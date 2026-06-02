use std::time::Duration;

pub(crate) const MAX_N: u16 = 255;
pub(crate) const BINOM_MAX: usize = 257;

pub static BINOM: [[f64; BINOM_MAX]; BINOM_MAX] = {
    let mut array = [[0.0; BINOM_MAX]; BINOM_MAX];
    let mut n = 0;
    while n < BINOM_MAX {
        array[n][0] = 1.0;
        let mut k = 1;
        while k <= n {
            array[n][k] = array[n - 1][k] + array[n - 1][k - 1];
            k += 1;
        }
        n += 1;
    }
    array
};

/// Probability mass function for the binomial distribution.
/// Assumes all arguments to be compatible with N_MAX and BINOM.
/// This has complexity of O(1)
#[inline(always)]
fn pmf(j: usize, n: usize, p_e_powers: &[f64], one_minus_p_e_powers: &[f64]) -> f64 {
    // Safety check for bounds
    if n >= BINOM_MAX || j > n {
        return 0.0;
    }

    // BINOM[n][j] * P_e^j * (1-P_e)^(n-j)
    BINOM[n][j] * p_e_powers[j] * one_minus_p_e_powers[n - j]
}

/// This has complexity of O(p) = O(min(k,p))
#[inline(always)]
pub fn calc_residual_erasure_rate_large_k(
    k: u16,
    p: u16,
    p_e: f64,
    ratio: f64,
    p_e_powers: &[f64],
    one_minus_p_e_powers: &[f64],
) -> f64 {
    if p_e >= 1.0 {
        return 1.0;
    }
    if p_e == 0.0 {
        return 0.0;
    }

    let k_usize = k as usize;
    let p_usize = p as usize;
    let mut ps_p: f64 = 0.0;
    let mut plr_1: f64 = 0.0;
    let min_pk = p.min(k) as usize;

    // Part 1: plr_1 computation (already O(p) - unchanged)
    for i in 0..=min_pk {
        plr_1 += BINOM[k_usize][i] * ps_p * i as f64;
        ps_p = ratio * (BINOM[p_usize][p_usize - i] + ps_p);
    }
    plr_1 *= p_e_powers[p_usize] * one_minus_p_e_powers[k_usize];

    let mut plr_2: f64 = 0.0;
    if k >= 1 {
        // Compute CDF: sum_{m=0 to p-1} C(k-1,m) * p_e^m * (1-p_e)^(k-1-m)
        let mut cdf_sum: f64 = 0.0;
        let upper_limit = p.min(k) as usize;

        for m in 0..upper_limit {
            cdf_sum +=
                BINOM[k_usize - 1][m] * p_e_powers[m] * one_minus_p_e_powers[k_usize - 1 - m];
        }

        // plr_2 = k * p_e * (1 - cdf_sum)
        plr_2 = k as f64 * p_e * (1.0 - cdf_sum);
    }

    (plr_1 + plr_2) / (k as f64)
}

/// This has complexity of O(k)
#[inline(always)]
pub fn calc_residual_erasure_rate_large_p(
    k: u16,
    p: u16,
    p_e: f64,
    ratio: f64, // p_e / (1-p_e) precomputed
    p_e_powers: &[f64],
    one_minus_p_e_powers: &[f64],
) -> f64 {
    if p_e >= 1.0 {
        return 1.0;
    }
    if p_e == 0.0 {
        return 0.0;
    }

    let k_usize = k as usize;
    let p_usize = p as usize;
    let mut ps_p: f64 = 0.0;
    let mut plr_1: f64 = 0.0;
    let min_pk = p.min(k) as usize;

    // Part 1: plr_1 computation with precomputed ratio
    for i in 0..=min_pk {
        plr_1 += BINOM[k_usize][i] * ps_p * i as f64;
        ps_p = ratio * (BINOM[p_usize][p_usize - i] + ps_p);
    }
    plr_1 *= p_e_powers[p_usize] * one_minus_p_e_powers[k_usize];

    // Part 2: plr_2 computation with precomputed ratio
    let mut plr_2: f64 = 0.0;
    if p <= k {
        let mut power_p_e = p_e_powers[p_usize] * one_minus_p_e_powers[k_usize - p_usize];

        for j in (p + 1)..=k {
            power_p_e *= ratio; // Use precomputed ratio instead of division
            plr_2 += j as f64 * BINOM[k_usize][j as usize] * power_p_e;
        }
    }

    (plr_1 + plr_2) / (k as f64)
}

/// Computes the probability of failing to decode cycle c.
/// This has complexiy O(k+c)
pub fn calculate_p_fail(
    c: usize,
    k: u16,
    np: &[u16],
    p_e_powers: &[f64],
    one_minus_p_e_powers: &[f64],
) -> f64 {
    // O(c)
    let cumulative_parity: u16 = np.iter().take(c + 1).sum();

    let n = (k + cumulative_parity) as usize;

    if n > MAX_N as usize {
        return 0.0;
    }

    // Failure Condition: Received count < k <--> Erasures > cumulative_parity_sum
    // We sum the probability of all erasure counts that lead to failure.
    // O(k)
    let mut failure_prob = 0.0;
    for erasures in (cumulative_parity + 1) as usize..=n {
        failure_prob += pmf(erasures, n, p_e_powers, one_minus_p_e_powers);
    }

    failure_prob
}

/// Computes the effective RI of a coding configuration in an IID channel with erasure rate p_e
/// This has complexity O(k*c + c^2)
pub fn get_effective_ri_arq(
    k: u16,
    n_p: &[u16],
    p_e_powers: &[f64],
    one_minus_p_e_powers: &[f64],
) -> f64 {
    let mut expected_parity = 0.0;

    for (i, &p_chunk) in n_p.iter().enumerate() {
        let prob_needed = if i == 0 {
            1.0
        } else {
            // Probability that previous rounds (0..i-1) failed
            calculate_p_fail(i - 1, k, n_p, p_e_powers, one_minus_p_e_powers)
        };

        expected_parity += p_chunk as f64 * prob_needed;
    }
    expected_parity / k as f64
}

#[derive(Debug, Clone)]
pub struct TimeStats {
    pub source_interval: Duration,
    pub channel_interval: Duration, // = P_L / R_C
    pub rtt: Duration,
    pub response_delay: Duration,
    pub receive_delay: Duration, // PRRT receiver protocol delay (aka how long does a packet need to get from socket rcv -> application)
    pub packet_loss_detection_delay: Duration,
    pub target_delay: Duration,
}

pub fn incremental_search<const NC_LIMIT: usize, const NC_LIMIT_PLUS1: usize>(
    channel_erasure_rate: f64,
    target_erasure_rate: f64,
    timestats: &TimeStats,
    max_kp: u16,
) -> Option<(u16, u16, [u16; NC_LIMIT_PLUS1], f64, f64)> {
    // compute n_c_max
    let fec_delay: Duration =
        (timestats.rtt).div_f32(2.) + timestats.response_delay + timestats.receive_delay;
    let n_c_max: usize = (timestats.target_delay.saturating_sub(fec_delay))
        .div_duration_f64(
            timestats.rtt + timestats.response_delay + timestats.packet_loss_detection_delay,
        )
        .floor() as usize;
    // 2. pick efficient subroutine
    if n_c_max == 0 || NC_LIMIT == 0 {
        incremental_search_fec::<NC_LIMIT, NC_LIMIT_PLUS1>(channel_erasure_rate, target_erasure_rate, timestats, max_kp)
    } else {
        incremental_search_arq::<NC_LIMIT, NC_LIMIT_PLUS1>(channel_erasure_rate, target_erasure_rate, timestats, max_kp)
    }
}

// ++++++++++++++++++++++++++++++++++++++++ //
// above is reference, below is optimized implementation. //
// ++++++++++++++++++++++++++++++++++++++++ //

pub fn incremental_search_fec<const NC_LIMIT: usize, const NC_LIMIT_PLUS1: usize>(
    channel_erasure_rate: f64,
    target_erasure_rate: f64,
    timestats: &TimeStats,
    max_kp: u16,
) -> Option<(u16, u16, [u16; NC_LIMIT_PLUS1], f64, f64)> {
    // 0. This is our base schedule k=1, p=0
    let mut base_k: u16 = 1;
    let mut base_p: u16 = 0;
    let mut erasure_rate: f64;
    let mut schedule_duration: Duration =
        (timestats.rtt).div_f32(2.) + timestats.response_delay + timestats.receive_delay;

    // 1. Trivial case: Channel is good enough without coding.
    // this implies we need redundancy -> p>0
    if channel_erasure_rate <= target_erasure_rate && schedule_duration <= timestats.target_delay {
        return Some((base_k, base_p, [0; NC_LIMIT_PLUS1], 0.0, channel_erasure_rate));
    }
    if schedule_duration > timestats.target_delay {
        return None;
    }

    // 2. precompute useful structures
    let p_e = channel_erasure_rate;
    let one_minus_p_e = 1.0 - p_e;
    let ratio = p_e / one_minus_p_e;

    let mut p_e_powers = [1.0; MAX_N as usize + 1];
    let mut one_minus_p_e_powers = [1.0; MAX_N as usize + 1];
    for i in 1..=MAX_N as usize {
        p_e_powers[i] = p_e_powers[i - 1] * p_e;
        one_minus_p_e_powers[i] = one_minus_p_e_powers[i - 1] * one_minus_p_e;
    }

    // 3. increment p until we find a schedule that satsifies the target erasure rate constraint.
    // This will ALWAYS result in p >= 1
    loop {
        base_p += 1;
        schedule_duration += timestats.channel_interval;
        if schedule_duration > timestats.target_delay {
            // delay constraint broken, didnt find any valid p yet so we are done, since we cannot do anything about it.
            return None;
        }
        if base_p > MAX_N || base_p > max_kp {
            return None;
        }
        erasure_rate = calc_residual_erasure_rate_large_p(
            1,
            base_p,
            p_e,
            ratio,
            &p_e_powers,
            &one_minus_p_e_powers,
        );
        if erasure_rate <= target_erasure_rate {
            break;
        }
    }

    // 4. if the base schedule is (k=1,p=1), we know that we need p/k = RI <= 1 -> we now increment k.
    if base_p == 1 {
        // Increment k until we find the maximum k where erasure_rate <= target with p=1
        loop {
            let next_k = base_k + 1;
            if next_k > max_kp || next_k + 1 > MAX_N {
                break;
            }

            erasure_rate = calc_residual_erasure_rate_large_k(
                next_k,
                1,
                p_e,
                ratio,
                &p_e_powers,
                &one_minus_p_e_powers,
            );

            if erasure_rate <= target_erasure_rate
                && schedule_duration + timestats.source_interval < timestats.target_delay
            {
                base_k = next_k;
                schedule_duration += timestats.source_interval;
            } else {
                break;
            }
        }
    }

    // 5. we now have the base schedule (base_k, base_p).
    // we can use the RI definition RI = p/k to define an upper bound for a k given p and p given k as a jump target.
    // This means that if k<p, we iterate over k and look for p that satisfy p_new/k_i < p_opt/k_opt.
    // Similarly if p<k, we iterate over p and look for k that satisfy p_i/k_new < p_opt/k_opt.
    // Note that while jumping is efficient, we have to iterate from the upper bound since tigher bounds likely exist.
    // This algorithm is optimal for fec.

    let mut k: u16 = base_k;
    let mut p: u16 = base_p;
    let mut best_k: u16 = base_k;
    let mut best_p: u16 = base_p;
    let mut best_ri: f64 = best_p as f64 / best_k as f64;

    if base_k <= base_p {
        // Optimized sub-routine for base_k <= base_p
        // iterate over k, and Find max p for k such that best_ri > p / k.
        schedule_duration =
            schedule_duration.saturating_sub(timestats.channel_interval.mul_f32(p as f32)); // We will add all parity delays manually.

        for k in best_k + 1..=(MAX_N - 1) {
            schedule_duration += timestats.source_interval;
            if k + p > MAX_N || k * p > max_kp {
                break;
            }

            // Find max p such that best_ri > p / k
            // => p < best_ri * prev_k
            p = ((best_ri * k as f64).ceil() as u16 - 1).min(MAX_N - k);

            erasure_rate = calc_residual_erasure_rate_large_p(
                k,
                p,
                p_e,
                ratio,
                &p_e_powers,
                &one_minus_p_e_powers,
            );

            // Check if this is a better solution
            if erasure_rate < target_erasure_rate {
                // Decrement p to explore lower values that might be better
                while p > 1 {
                    p -= 1;
                    erasure_rate = calc_residual_erasure_rate_large_p(
                        k,
                        p,
                        p_e,
                        ratio,
                        &p_e_powers,
                        &one_minus_p_e_powers,
                    );

                    if erasure_rate > target_erasure_rate {
                        p += 1;
                        break;
                    }
                }
                let delay = schedule_duration + timestats.channel_interval.mul_f32(p as f32);
                if delay > timestats.target_delay {
                    // We are done. This works because we have no other options to make this better. they were already considered.
                    break;
                }
                if k * p > max_kp {
                    break;
                }
                best_k = k;
                best_p = p;
                best_ri = p as f64 / k as f64;
            }
        }
    } else {
        // Optimized sub-routine for base_p < base_k
        // iterate over p, and Find max k for p such that best_ri > p / k.
        schedule_duration =
            schedule_duration.saturating_sub(timestats.source_interval.mul_f32(k as f32)); // We will add all source delays manually.

        for p in best_p + 1..=MAX_N - 1 {
            if k + p > MAX_N || k * p > max_kp {
                break;
            }
            // Find max k such that best_ri > p / k
            // => k > p / best_ri
            k = ((p as f64 / best_ri).floor() as u16 + 1)
                .min(MAX_N - p)
                .max(1);
            if k * p > max_kp {
                break;
            }
            erasure_rate = calc_residual_erasure_rate_large_k(
                k,
                p,
                p_e,
                ratio,
                &p_e_powers,
                &one_minus_p_e_powers,
            );

            // Check if this is a better solution
            if erasure_rate < target_erasure_rate {
                // Increment k to explore higher values that might be better
                while k + p <= MAX_N {
                    k += 1;

                    if k + p > MAX_N || k * p > max_kp {
                        k -= 1;
                        break;
                    }

                    erasure_rate = calc_residual_erasure_rate_large_k(
                        k,
                        p,
                        p_e,
                        ratio,
                        &p_e_powers,
                        &one_minus_p_e_powers,
                    );

                    if erasure_rate > target_erasure_rate {
                        k -= 1;
                        break;
                    }
                }
                let delay = schedule_duration + timestats.source_interval.mul_f32(k as f32);
                if delay > timestats.target_delay {
                    // We are done. This works because we have no other options to make this better. they were already considered.
                    break;
                }
                best_k = k;
                best_p = p;
                best_ri = p as f64 / k as f64;
            }
        }
    }

    //dbg!((best_k, best_p));
    let mut schedule = [0; NC_LIMIT_PLUS1];
    schedule[0] = best_p;
    let best_erasure_rate: f64 = if k > p {
        calc_residual_erasure_rate_large_k(k, p, p_e, ratio, &p_e_powers, &one_minus_p_e_powers)
    } else {
        calc_residual_erasure_rate_large_p(k, p, p_e, ratio, &p_e_powers, &one_minus_p_e_powers)
    };
    Some((best_k, best_p, schedule, best_ri, best_erasure_rate))
}

/// Incremental search for the optimal ARQ schedule.
///
/// The ARQ case is significantly more complex than the FEC case.
/// We have to search for k, p and a new N_P, the ARQ schedule.
/// The introduction of N_P has no impact on the erasure rate constraint.
/// N_P however has significant impact on the RI computation.
/// Rounds in N_P act as incremental redundancy, which results in a roughly exponential decrease in RI.
/// There are some properties of N_P that we can exploit to arrive at an efficient solution:
/// 1. The ARQ cycles (c>0) of an optimal N_P schedule are an ascending integer composition.
/// 2. In practice, the search is used in applications with a deadline constraint with a sense of urgency.
///     - Thus, we will limit the maximum number of ARQ cycles by a constant N_C_MAX = 4.
///     - This constant can be changed to either speed up the computation or increase its quality.
///     - With an optimized N_C_MAX = 4, we still benefit from the exponential decrease in RI compared to FEC.
///     - Note that the exponential decay in RI is bounded. simply adding more and more cycles has a diminishing effect.
///         - This depends e.g. on the erasure rate and variance of our underlying IID channel. (maybe higher moments as well)
///         - Think about he relative difference a parity packet makes. if p_e=0.01 and k=100 and p=2:
///             - These two packets have MASSIVE exponential decay in their cycles.
///             - for high erasure rates, this becomes less pronounced, which is why we have to bundle parity packets in shorter N_Cs
///     - We will implement the incremental_search_arq const generic over N_C_MAX.
///     - increasing N_C_MAX may introduce exponential computational overhead.
/// 3. With these restrictions, we will no longer produce optimal results, however they remain deterministic, high quality and general.
/// 4. For environments that are bounded by N_C_MAX, the search is near-optimal.
/// 5. The algorithm is aware of the schedule duration and continously operates below the target_delay.
///     - We use an early-stop heuristic and prevent searching after one cycle reduction in the main loop.
///     - the valid base configuration does not use this heuristic. only the main loop does.
/// 6. The algorithm is aware of the platform constraints defined as max_kp -> k*p <= max_kp.
/// 7. We use a `greedy_add_parity` approach to incrementally construct the optimal schedule
///     - This is an approximation. I think. It performs exceptionally well for now.
///     - Currently i do not see the value of making this more complex, aka more optimal.
///     - if the need arises, you may want to improve the performance as follos:
///       a. find and remove the parity packet that increases the RI the most of the current schedule
///       b. re-insert it by checking the potential RI of every valid (ascending) slot.
///       c. insert the new parity packet like in b.
///     - Scaling the amount of re-inserts will get you really far.
///
/// Basic idea is we iterate over n (until max_n), and try satisfying the loss constraint. If the previous iteration was valid, we
/// increment k, since valid configs dont need more parity.
/// Otherwise, we increment p, since invalid configs need more parity.
/// While doing that we keep track of the target_delay (timestats) and platform constraints (max_kp).
pub fn incremental_search_arq<const NC_LIMIT: usize, const NC_LIMIT_PLUS1: usize>(
    channel_erasure_rate: f64,
    target_erasure_rate: f64,
    timestats: &TimeStats,
    max_kp: u16,
) -> Option<(u16, u16, [u16; NC_LIMIT_PLUS1], f64, f64)> {
    let k1_delay = (timestats.rtt).div_f32(2.) + timestats.response_delay + timestats.receive_delay;
    // 1. Trivial case: Channel is good enough without coding.
    if channel_erasure_rate <= target_erasure_rate {
        if k1_delay <= timestats.target_delay {
            return Some((1, 0, [0; NC_LIMIT_PLUS1], 0.0, channel_erasure_rate));
        } else {
            return None;
        }
    }

    // 2. Precompute structures
    let p_e = channel_erasure_rate;
    let one_minus_p_e = 1.0 - p_e;
    let ratio = if one_minus_p_e < 1e-9 {
        return None; // if we only get one in 10^9 packets, we give up.
    } else {
        p_e / one_minus_p_e
    };

    let mut p_e_powers = [1.0; MAX_N as usize + 1];
    let mut one_minus_p_e_powers = [1.0; MAX_N as usize + 1];
    for i in 1..=MAX_N as usize {
        p_e_powers[i] = p_e_powers[i - 1] * p_e;
        one_minus_p_e_powers[i] = one_minus_p_e_powers[i - 1] * one_minus_p_e;
    }

    // --- Helper Closures ---

    let greedy_add_parity = |k: u16,
                             schedule: &mut [u16; NC_LIMIT_PLUS1],
                             schedule_len: &mut usize,
                             schedule_max_len: usize,
                             schedule_duration: &mut Duration| {
        // 1. Adding a new cycle is always optimal.
        if *schedule_len < schedule_max_len {
            *schedule_len += 1; // cycle 0 is proactive, we want to insert the packet into the NEXT cycle, so increment first.
            schedule[*schedule_len] = 1;
            *schedule_duration += timestats.packet_loss_detection_delay
                + timestats.rtt
                + timestats.response_delay
                + timestats.channel_interval;
            return;
        }
        // 2. Find the position to incrementing that has the lowest RI increase.
        let mut min_ri = f64::MAX;
        let mut best_cycle = 0;

        // Iterate including schedule_len, due to existence of FEC cycle.
        for i in 0..=*schedule_len {
            // ARQ cycles are always ascending, except proactive schedule[0].
            if (i + 1 < *schedule_len && schedule[i] + 1 > schedule[i + 1]) && i != 0 {
                continue;
            }
            schedule[i] += 1;
            let ri = get_effective_ri_arq(
                k,
                &schedule[..=*schedule_len],
                &p_e_powers,
                &one_minus_p_e_powers,
            );
            schedule[i] -= 1; // Restore

            if ri < min_ri {
                min_ri = ri;
                best_cycle = i;
            }
        }

        // 3. Apply the increment.
        schedule[best_cycle] += 1;
        *schedule_duration += timestats.channel_interval;
    };

    // 3. Initial step.
    let mut curr_k: u16 = 1;
    let mut curr_p: u16 = 0;
    let mut curr_schedule: [u16; NC_LIMIT_PLUS1] = [0; NC_LIMIT_PLUS1];
    let mut curr_cycle_count: usize = 0; // the proactive cycle always exist, and is initially set to 0.
    let mut schedule_max_cycles = NC_LIMIT; // This will be decremented at most once. note that schedule_max_cycles is a valid index.
    // The initial duration DOES NOT include the source packet delay, since the duration of the schedule starts with the first source packet.
    let mut schedule_duration: Duration = k1_delay;

    // Increment p until valid.
    loop {
        let residual = calc_residual_erasure_rate_large_p(
            curr_k,
            curr_p,
            p_e,
            ratio,
            &p_e_powers,
            &one_minus_p_e_powers,
        );

        if residual <= target_erasure_rate {
            // Found valid starting point
            break;
        }

        // Increment p and update schedule
        curr_p += 1;
        if curr_p > MAX_N {
            return None;
        }
        greedy_add_parity(
            curr_k,
            &mut curr_schedule,
            &mut curr_cycle_count,
            schedule_max_cycles,
            &mut schedule_duration,
        );
        // This is the special case: we will aggressively remove cycles until we find at least one valid configuration.
        while schedule_duration > timestats.target_delay && schedule_max_cycles != 0 {
            let redistribution_count = curr_schedule[curr_cycle_count]; // take parity from last active cycle
            curr_schedule[curr_cycle_count] = 0; // remove parity from last active cycle
            curr_cycle_count -= 1; // decrement curr_cycle_count
            schedule_max_cycles = curr_cycle_count; // dont allow the cycle to be created again
            // Adjust the schedule duration, one less cycle and redistribution_count less parity packets
            schedule_duration = schedule_duration.saturating_sub(
                timestats.packet_loss_detection_delay
                    + timestats.rtt
                    + timestats.response_delay
                    + timestats
                        .channel_interval
                        .mul_f64(redistribution_count as f64),
            );
            // Redistribute, this adds back the parity packet delays
            for _ in 0..redistribution_count {
                greedy_add_parity(
                    curr_k,
                    &mut curr_schedule,
                    &mut curr_cycle_count,
                    schedule_max_cycles,
                    &mut schedule_duration,
                );
            }
        }
        // If we fail to fulfil the delay constraint, no solution exists
        if schedule_duration > timestats.target_delay {
            return None;
        }
    }

    // 4. Main Search Loop
    let mut opt_k = curr_k;
    let mut opt_p = curr_p;
    let mut opt_schedule = curr_schedule;
    let mut _opt_schedule_len = curr_cycle_count;
    let mut opt_ri = get_effective_ri_arq(
        curr_k,
        &curr_schedule[..=curr_cycle_count],
        &p_e_powers,
        &one_minus_p_e_powers,
    );

    let mut previous_config_valid: bool = true;
    let mut did_previously_truncate: bool = false;

    for _n in (curr_k + curr_p) + 1..=MAX_N {
        if previous_config_valid {
            // Increment k
            curr_k += 1;
            schedule_duration += timestats.source_interval;
        } else {
            // Increment p
            curr_p += 1;
            greedy_add_parity(
                curr_k,
                &mut curr_schedule,
                &mut curr_cycle_count,
                schedule_max_cycles,
                &mut schedule_duration,
            );
        }
        if curr_k * curr_p > max_kp {
            break;
        }
        // Check delay constraint, truncate if not.
        // We will do this at most once. removing cycles generally degrades RI exponentially.
        if schedule_duration > timestats.target_delay {
            if did_previously_truncate || schedule_max_cycles == 0 {
                // We either did redistribute once, in which case doing it again gives (VERY LIKELY?) no benefit.
                // Or we cannot reduce the cycle count any more
                break;
            }
            did_previously_truncate = true;
            let redistribution_count = curr_schedule[curr_cycle_count]; // take parity from last active cycle
            curr_schedule[curr_cycle_count] = 0; // remove parity from last active cycle
            curr_cycle_count -= 1; // decrement curr_cycle_count
            schedule_max_cycles = curr_cycle_count; // dont allow the cycle to be created again
            // Adjust the schedule duration, one less cycle and redistribution_count less parity packets
            schedule_duration = schedule_duration.saturating_sub(
                timestats.packet_loss_detection_delay
                    + timestats.rtt
                    + timestats.response_delay
                    + timestats
                        .channel_interval
                        .mul_f64(redistribution_count as f64),
            );
            // Redistribute, this adds back the parity packet delays
            for _ in 0..redistribution_count {
                greedy_add_parity(
                    curr_k,
                    &mut curr_schedule,
                    &mut curr_cycle_count,
                    schedule_max_cycles,
                    &mut schedule_duration,
                );
            }
            if schedule_duration > timestats.target_delay {
                // Redistribution did not reduce schedule_duration -> we are done
                break;
            }
        }

        let residual_erasure_rate = if curr_k >= curr_p {
            calc_residual_erasure_rate_large_k(
                curr_k,
                curr_p,
                p_e,
                ratio,
                &p_e_powers,
                &one_minus_p_e_powers,
            )
        } else {
            calc_residual_erasure_rate_large_p(
                curr_k,
                curr_p,
                p_e,
                ratio,
                &p_e_powers,
                &one_minus_p_e_powers,
            )
        };

        if residual_erasure_rate <= target_erasure_rate {
            previous_config_valid = true;
            let current_ri = get_effective_ri_arq(
                curr_k,
                &curr_schedule[..=curr_cycle_count],
                &p_e_powers,
                &one_minus_p_e_powers,
            );

            if current_ri < opt_ri {
                opt_ri = current_ri;
                opt_k = curr_k;
                opt_p = curr_p;
                opt_schedule = curr_schedule;
                _opt_schedule_len = curr_cycle_count;
            }
        } else {
            previous_config_valid = false;
        }
    }

    let opt_erasure_rate: f64 = if opt_k > opt_p {
        calc_residual_erasure_rate_large_k(
            opt_k,
            opt_p,
            p_e,
            ratio,
            &p_e_powers,
            &one_minus_p_e_powers,
        )
    } else {
        calc_residual_erasure_rate_large_p(
            opt_k,
            opt_p,
            p_e,
            ratio,
            &p_e_powers,
            &one_minus_p_e_powers,
        )
    };
    Some((opt_k, opt_p, opt_schedule, opt_ri, opt_erasure_rate))
}