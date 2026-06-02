pub mod incremental_search;
pub mod incremental_search_full;

use rand::rng;
use rand::rngs::ThreadRng;
use rand::seq::IndexedRandom;

use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use std::time::Instant;
use std::io::BufWriter;

use rand::Rng;

use rayon::prelude::*;

use crate::incremental_search::TimeStats;
use crate::incremental_search::incremental_search;
use crate::incremental_search_full::incremental_search_full;

#[derive(Debug, Clone)]
pub struct SearchInput {
    pub channel_erasure_rate: f64,
    pub target_erasure_rate: f64,
    pub timestats: TimeStats,
    pub max_kp: u16,
}

/// Adapted from "Pablo Gil Pereira. Predictable Data Transport: A Delay and Energy Perspective." Table 3.2
/// `gen_val` explicitly generates floating point numbers.
fn generate_random_input(rng: &mut ThreadRng, max_kp: u16) -> SearchInput {
    // Constant Timestats
    let channel_interval = Duration::ZERO;
    let response_delay = Duration::from_micros(200);
    let receive_delay = Duration::from_micros(200);

    // Helper closure to implement the specific logic:
    // 1. Pick magnitude from slice
    // 2. Generate float [1.0, 10.0)
    // 3. Multiply
    let mut gen_val = |magnitudes: &[f64]| -> f64 {
        let magnitude = *magnitudes.choose(rng).unwrap();
        let multiplier = rng.random_range(1.0..10.0);
        magnitude * multiplier
    };

    // 1. RTT
    // Magnitudes: 1ms, 10ms, 100ms
    let rtt_secs = gen_val(&[0.001, 0.01, 0.1]);
    let rtt = Duration::from_secs_f64(rtt_secs);
    let packet_loss_detection_delay = rtt.mul_f64(0.2);

    // 2. Target delay
    // Magnitudes: 1ms, 10ms, 100ms
    let target_delay = Duration::from_secs_f64(gen_val(&[0.001, 0.01, 0.1]));

    // 3. Source packet interval
    // Magnitudes: 0.1ms, 1ms, 10ms
    let source_interval = Duration::from_secs_f64(gen_val(&[0.0001, 0.001, 0.01]));

    // 4. Target PLR
    // Magnitudes: 1e-3, 1e-4, 1e-5
    let target_erasure_rate = gen_val(&[1e-3, 1e-4, 1e-5]);

    // 5. Channel erasure rate
    // Magnitudes: 1e-1, 1e-2, 1e-3, 1e-4
    let channel_erasure_rate = gen_val(&[1e-1, 1e-2, 1e-3, 1e-4]);

    let timestats = TimeStats {
        source_interval,
        channel_interval,
        rtt,
        response_delay,
        receive_delay,
        packet_loss_detection_delay,
        target_delay,
    };

    SearchInput {
        channel_erasure_rate,
        target_erasure_rate,
        timestats,
        max_kp,
    }
}


fn run_benchmark_for_config<const NC: usize, const NC_PLUS1: usize>(max_kp: u16) {
    const N_SAMPLES: usize = 1_000_000;
    const N_RUNS_PER_SAMPLE: u32 = 10;

    let mut rng: ThreadRng = rng();
    let mut averages: Vec<Duration> = Vec::with_capacity(N_SAMPLES);

    for _ in 0..N_SAMPLES {
        let input: SearchInput = generate_random_input(&mut rng, max_kp);

        // Check config validity with a single run using the const generic NC limit
        let valid = incremental_search::<NC, NC_PLUS1>(
            input.channel_erasure_rate,
            input.target_erasure_rate,
            &input.timestats,
            input.max_kp,
        )
        .is_some();

        if !valid {
            continue;
        }

        // Time 10 runs of incremental_search for this config
        let start = Instant::now();
        for _ in 0..N_RUNS_PER_SAMPLE {
            let _ = incremental_search::<NC, NC_PLUS1>(
                input.channel_erasure_rate,
                input.target_erasure_rate,
                &input.timestats,
                input.max_kp,
            );
        }
        let total_elapsed = start.elapsed();

        // Compute average duration per run
        let avg_nanos = total_elapsed.as_nanos() / (N_RUNS_PER_SAMPLE as u128);
        let avg_duration = Duration::from_nanos(avg_nanos as u64);

        averages.push(avg_duration);
    }

    // Write results to CSV file.
    let filename: String = format!("../python/inc_search_eval/results_execution_time_NCL{}_maxkp{}.csv.br", NC, max_kp);
    let file = File::create(&filename).unwrap_or_else(|_| panic!("Failed to create {}", filename));
    let buf = BufWriter::new(file);
    let mut writer = brotli::CompressorWriter::new(buf, 4096, 8, 22);
    writeln!(writer, "execution_time_ns").unwrap();
    for d in averages {
        writeln!(writer, "{}", d.as_nanos()).unwrap();
    }
    println!("Saved benchmark results to {}", filename);
}

pub fn benchmark_incremental_search_averages() {
    run_benchmark_for_config::<0,{0+1}>(2048);
    run_benchmark_for_config::<5,{5+1}>(256);
    run_benchmark_for_config::<5,{5+1}>(2048);
    run_benchmark_for_config::<5,{5+1}>(65535);
    run_benchmark_for_config::<9,{9+1}>(2048);
}

// Per-worker statistics
#[derive(Clone, Copy, Debug, Default)]
struct WorkerStats {
    valid_comparisons: u64,
    invalid_configs: u64,
    deviations: u64,
    total_diff: f64,
    diff_sum_when_deviated: f64,
    full_better_count: u64,
    greedy_better_count: u64,
}

impl WorkerStats {
    fn accumulate(self, other: WorkerStats) -> WorkerStats {
        WorkerStats {
            valid_comparisons: self.valid_comparisons + other.valid_comparisons,
            invalid_configs: self.invalid_configs + other.invalid_configs,
            deviations: self.deviations + other.deviations,
            total_diff: self.total_diff + other.total_diff,
            diff_sum_when_deviated: self.diff_sum_when_deviated + other.diff_sum_when_deviated,
            full_better_count: self.full_better_count + other.full_better_count,
            greedy_better_count: self.greedy_better_count + other.greedy_better_count,
        }
    }
}

fn run_nc_eval_for_limits<const FULL_NC: usize, const FULL_NC_PLUS1: usize, const GREEDY_NC: usize, const GREEDY_NC_PLUS1: usize>(writer: &mut impl Write) {
    let target_valid: u64 = 2_500_000;
    
    let n_workers = rayon::current_num_threads();
    let base_per_worker = target_valid / n_workers as u64;
    let remainder = target_valid % n_workers as u64;

    let stats = (0..n_workers)
        .into_par_iter()
        .map(|worker_id| {
            let mut rng: ThreadRng = rng();
            let mut s = WorkerStats::default();

            // Add any remainder to the first worker to guarantee exactly target_valid
            let worker_target = base_per_worker + if worker_id == 0 { remainder } else { 0 };

            while s.valid_comparisons < worker_target {
                let input = generate_random_input(&mut rng, 2048);

                // Run greedy
                let res_greedy = incremental_search::<GREEDY_NC, GREEDY_NC_PLUS1>(
                    input.channel_erasure_rate,
                    input.target_erasure_rate,
                    &input.timestats,
                    input.max_kp,
                );

                // Run full
                let res_full = incremental_search_full::<FULL_NC, FULL_NC_PLUS1>(
                    input.channel_erasure_rate,
                    input.target_erasure_rate,
                    &input.timestats,
                    input.max_kp,
                );

                match (res_greedy, res_full) {
                    (Some((_k1, _p1, _, ri1, _)), Some((_k2, _p2, _, ri2, _))) => {
                        s.valid_comparisons += 1;

                        let diff = (ri1 - ri2).abs();
                        s.total_diff += diff;

                        if diff > 1e-9 {
                            s.deviations += 1;
                            s.diff_sum_when_deviated += diff;

                            if ri2 < ri1 {
                                s.full_better_count += 1;
                            } else {
                                s.greedy_better_count += 1;
                            }
                        }
                    }
                    _ => {
                        s.invalid_configs += 1;
                    }
                }
            }

            s
        })
        .reduce(WorkerStats::default, WorkerStats::accumulate);

    let avg_diff_total = if stats.valid_comparisons > 0 {
        stats.total_diff / stats.valid_comparisons as f64
    } else {
        0.0
    };

    let avg_diff_deviated = if stats.deviations > 0 {
        stats.diff_sum_when_deviated / stats.deviations as f64
    } else {
        0.0
    };

    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{}",
        FULL_NC,
        GREEDY_NC,
        stats.valid_comparisons,
        stats.invalid_configs,
        stats.deviations,
        avg_diff_total,
        avg_diff_deviated,
        stats.full_better_count,
        stats.greedy_better_count
    )
    .unwrap();
    
    println!("Finished evaluation for FULL_NC={} GREEDY_NC={}", FULL_NC, GREEDY_NC);
}

fn nc_limit_eval() {
    let file = File::create("../python/inc_search_eval/results_nc_limits.csv.br").expect("Failed to create results_nc_limits.csv.br");
    let buf = BufWriter::new(file);
    let mut writer = brotli::CompressorWriter::new(buf, 4096, 8, 22);
    writeln!(
        writer,
        "full_nc_limit,greedy_nc_limit,valid_configs,invalid_configs,deviations,total_avg_deviation,avg_deviation,full_better,greedy_better"
    )
    .unwrap();

    
    // GREEDY 0
    run_nc_eval_for_limits::<0, {0+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<1, {1+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<2, {2+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<3, {3+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<4, {4+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<5, {5+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<6, {6+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},0, {0+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},0, {0+1}>(&mut writer);

    // GREEDY 1
    run_nc_eval_for_limits::<1, {1+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<2, {2+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<3, {3+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<4, {4+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<5, {5+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<6, {6+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},1, {1+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},1, {1+1}>(&mut writer);

    // GREEDY 2
    run_nc_eval_for_limits::<2, {2+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<3, {3+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<4, {4+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<5, {5+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<6, {6+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},2, {2+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},2, {2+1}>(&mut writer);

    // GREEDY 3
    run_nc_eval_for_limits::<3, {3+1},3, {3+1}>(&mut writer);
    run_nc_eval_for_limits::<4, {4+1},3, {3+1}>(&mut writer);
    run_nc_eval_for_limits::<5, {5+1},3, {3+1}>(&mut writer);
    run_nc_eval_for_limits::<6, {6+1},3, {3+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},3, {3+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},3, {3+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},3, {3+1}>(&mut writer);

    // GREEDY 4
    run_nc_eval_for_limits::<4, {4+1},4, {4+1}>(&mut writer);
    run_nc_eval_for_limits::<5, {5+1},4, {4+1}>(&mut writer);
    run_nc_eval_for_limits::<6, {6+1},4, {4+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},4, {4+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},4, {4+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},4, {4+1}>(&mut writer);

    // GREEDY 5
    run_nc_eval_for_limits::<5, {5+1},5, {5+1}>(&mut writer);
    run_nc_eval_for_limits::<6, {6+1},5, {5+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},5, {5+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},5, {5+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},5, {5+1}>(&mut writer);

    // GREEDY 6
    run_nc_eval_for_limits::<6, {6+1},6, {6+1}>(&mut writer);
    run_nc_eval_for_limits::<7, {7+1},6, {6+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},6, {6+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},6, {6+1}>(&mut writer);

    // GREEDY 7
    run_nc_eval_for_limits::<7, {7+1},7, {7+1}>(&mut writer);
    run_nc_eval_for_limits::<8, {8+1},7, {7+1}>(&mut writer);
    run_nc_eval_for_limits::<9, {9+1},7, {7+1}>(&mut writer);
}


fn main() {
    fs::create_dir_all("../python/inc_search_eval")
        .expect("Failed to create output directory ../python/inc_search_eval");
    benchmark_incremental_search_averages();
    nc_limit_eval();
}
