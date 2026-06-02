use caps::{CapSet, Capability};
use clap::Parser;
use lltc::{DelayConfig, LossConfig, NetworkConfig, NetworkConfigurationScheduler};
use prrt::types::protocol_time::ProtocolTime;
use prrt::Sock;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::{
    process::exit,
    sync::{Arc, Barrier},
    thread::{self},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const ENABLE_LOGGING: bool = false;
const SENDER_PORT: u16 = 4820;
const RECEIVER_PORT: u16 = 4821;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PhaseConfig {
    loss_rate: f64,
    correlation: f64,
    rtt_ms: u64,
    target_erasure_rate: f64,
    packet_count: usize,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// JSON string containing list of PhaseConfig objects
    #[arg(long)]
    config_json: String,

    /// Initial sender target delay in ms
    #[arg(long, default_value_t = 200)]
    sender_delay_ms: u64,

    /// Inter-packet delay in ms (pacing)
    #[arg(long, default_value_t = 1)]
    inter_packet_delay_us: u64,
}

#[inline(always)]
fn spin_sleep_until(deadline: Instant) {
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

fn main() {
    let base_message = "Hello PRRT! from eva";

    {
        let sample_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let sample_message = format!("{}|{:09}|{}", base_message, 0, sample_ts);
        eprintln!("Packet length: {} bytes", sample_message.as_bytes().len());
    }

    // 1. Initialize Capabilities
    if let Err(e) = enable_ambient_capabilities() {
        eprintln!("Capability Initialization Failed: {}", e);
        std::process::exit(1);
    }

    // 2. Parse Arguments and Config
    let args = Args::parse();
    let phases: Vec<PhaseConfig> = serde_json::from_str(&args.config_json)
        .expect("Failed to parse config_json. Ensure it is valid JSON.");

    if phases.is_empty() {
        eprintln!("Error: No phases defined in configuration.");
        exit(1);
    }

    let total_messages: usize = phases.iter().map(|p| p.packet_count).sum();
    let initial_sender_target_delay = Duration::from_millis(args.sender_delay_ms);
    let inter_packet_delay = Duration::from_micros(args.inter_packet_delay_us);

    // 3. Setup Network Config (TC)
    let mut tc_scheduler = create_tc_scheduler(&phases);

    // 4. Setup Sockets
    let sender_addr = format!("127.0.0.1:{}", SENDER_PORT).parse().unwrap();
    let receiver_addr = format!("127.0.0.1:{}", RECEIVER_PORT).parse().unwrap();

    let mut sender: Sock = Sock::new_sender(
        sender_addr,
        receiver_addr,
        phases[0].target_erasure_rate,
        initial_sender_target_delay,
    )
    .expect("Failed to create sender socket!");

    let mut receiver: Sock = Sock::new_receiver(receiver_addr, sender_addr).unwrap();

    // 5. Receiver Thread
    let barrier = Arc::new(Barrier::new(2));
    let barrier_clone = Arc::clone(&barrier);

    let _receiver_thread = thread::spawn(move || {
        barrier_clone.wait();
        let mut received = 0;
        let mut total_latency_ns = 0u128;

        let mut last_receive_time = Instant::now();
        let timeout_duration = Duration::from_secs(5);

        loop {
            match receiver.receive_with_timeout(Duration::from_nanos(1)) {
                Ok(buf) => {
                    last_receive_time = Instant::now(); // Reset timeout timer

                    if ENABLE_LOGGING {
                        let received_message = match String::from_utf8(buf) {
                            Ok(s) => s,
                            Err(e) => {
                                eprintln!("{e}",);
                                continue;
                            }
                        };

                        let split: Vec<&str> = received_message.split('|').collect();
                        if split.len() >= 3 {
                            let index = split[1];
                            let sent_timestamp_ns: u128 = split[2].parse().unwrap_or(0);
                            let current_timestamp_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos();

                            let elapsed_ns = current_timestamp_ns.saturating_sub(sent_timestamp_ns);
                            let elapsed_ms = elapsed_ns as f64 / 1_000_000.0;

                            total_latency_ns += elapsed_ns;
                            received += 1;
                            let avg = (total_latency_ns as f64 / received as f64) / 1_000_000.0;

                            println!("APP Received {index:06} {received:06}th delay={elapsed_ms:.4}, avg={avg:.4}");
                        }
                    }
                }
                Err(_) => {
                    // Check if we should exit due to timeout
                    if Instant::now().duration_since(last_receive_time) > timeout_duration {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }
    });

    barrier.wait();

    // --- SETUP PHASE TIMING LOG ---
   // --- SETUP PHASE TIMING LOG ---
    let trace_dir_str = std::env::var("PRRT_TRACE_DIR").unwrap_or_else(|_| "traces".to_string());
    let trace_dir = std::path::Path::new(&trace_dir_str);

    // Create the directory (and any necessary parent directories)
    create_dir_all(trace_dir).unwrap_or_else(|e| {
        panic!("Failed to create trace directory '{}': {}", trace_dir_str, e);
    });

    // Safely append the filename to the directory path
    let phase_log_path = trace_dir.join("phase_switches.csv");
    let mut phase_log = File::create(&phase_log_path)
        .unwrap_or_else(|e| panic!("Failed to create {:?}: {}", phase_log_path, e));

    // 6. Apply Initial Network Config (Phase 0)
    tc_scheduler.apply_next().unwrap();
    if ENABLE_LOGGING {
        println!("Applied Phase 0 Config: {:?}", phases[0]);
    }
    let ts = ProtocolTime::now_raw_nanos();
    writeln!(phase_log, "{}", ts).expect("Failed to write phase log");

    let mut current_phase_idx = 0;
    let mut next_phase_threshold = phases[0].packet_count;
    
    // 7. Send Loop (Busy Wait)
    let start_time = Instant::now();
    let mut next_send_time = start_time;

    for i in 0..total_messages {
        if i == next_phase_threshold && current_phase_idx < phases.len() - 1 {
            current_phase_idx += 1;
            let config = &phases[current_phase_idx];

            tc_scheduler.apply_next().unwrap();
            sender.update_constraints(config.target_erasure_rate, initial_sender_target_delay);

            // --- LOG PHASE SWITCH ---
            let ts = ProtocolTime::now_raw_nanos();
            writeln!(phase_log, "{}", ts).expect("Failed to write phase log");

            if ENABLE_LOGGING {
                println!("--- Switching to Phase {} ---", current_phase_idx);
                println!("Applied Config: {:?}", config);
            }

            next_phase_threshold += config.packet_count;
        }

        let send_timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let message_content = format!("{}|{:09}|{}", base_message, i, send_timestamp_ns);

        // Wait until exact send time
        spin_sleep_until(next_send_time);

        sender.send(message_content.as_bytes().to_vec());

        // Schedule next packet
        next_send_time += inter_packet_delay;
    }

    // Brief wait to allow final packets to arrive (using spin for consistency, though sleep is fine here)
    spin_sleep_until(Instant::now() + Duration::from_secs(2));

    drop(tc_scheduler);
    sender.close();
    exit(0);
}

fn create_tc_scheduler(phases: &[PhaseConfig]) -> NetworkConfigurationScheduler {
    let mut scheduler = NetworkConfigurationScheduler::new();
    let to_gilbert_params = |rate: f64, rho: f64| -> (f32, f32) {
        let rho = rho.clamp(0.0, 0.99);
        let p_ge = rate * (1.0 - rho);
        let r_ge = (1.0 - rate) * (1.0 - rho);
        (p_ge as f32, r_ge as f32)
    };

    for phase in phases {
        let (p, r) = to_gilbert_params(phase.loss_rate, phase.correlation);
        let half_rtt = phase.rtt_ms / 2;

        let config = NetworkConfig {
            device: "lo".to_string(),
            sender_port: Some(SENDER_PORT),
            receiver_port: Some(RECEIVER_PORT),
            delay: DelayConfig {
                time: Duration::from_millis(half_rtt),
                ..Default::default()
            },
            loss: LossConfig {
                kind: lltc::LossKind::GilbertElliot {
                    p,
                    r,
                    h: 0.0,
                    k: 1.0,
                },
                ..Default::default()
            },
        };
        scheduler.queue(config);
    }

    scheduler
}

fn enable_ambient_capabilities() -> Result<(), Box<dyn Error>> {
    let required_caps = vec![Capability::CAP_NET_ADMIN, Capability::CAP_SYS_NICE];
    for cap in required_caps {
        if !caps::has_cap(None, CapSet::Permitted, cap)? {
            return Err(format!("Missing capability {:?}.", cap).into());
        }
        let mut inheritable = caps::read(None, CapSet::Inheritable)?;
        inheritable.insert(cap);
        caps::set(None, CapSet::Inheritable, &inheritable)?;
        let mut ambient = caps::read(None, CapSet::Ambient)?;
        ambient.insert(cap);
        if let Err(_) = caps::set(None, CapSet::Ambient, &ambient) {
            // warn but ignore
        }
    }
    Ok(())
}
