from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
import brotli
import io
import subprocess
import numpy as np
import matplotlib.lines as mlines

PHASE_LABEL_VERTICAL_OFFSET = 1.32

def parse_time_to_us(t_val):
    """Parses a time string or number into microseconds."""
    if isinstance(t_val, str):
        t_str = t_val.strip()
        if t_str.endswith('µs') or t_str.endswith('us'):
            return float(t_str[:-2])
        elif t_str.endswith('ns'):
            return float(t_str[:-2]) / 1000.0
        elif t_str.endswith('ms'):
            return float(t_str[:-2]) * 1000.0
        elif t_str.endswith('s'):
            return float(t_str[:-1]) * 1e6
            
    # If it has no suffix, or is already a raw number, treat it as nanoseconds
    try:
        return float(t_val) / 1000.0
    except ValueError:
        return np.nan

def read_brotli_csv(path, **kwargs):
    """
    Reads a Brotli-compressed CSV file by decompressing it into memory first.
    Accepts **kwargs to pass arguments like nrows to pd.read_csv.
    """
    try:
        with open(path, 'rb') as f:
            compressed_data = f.read()
        decompressed_data = brotli.decompress(compressed_data)
        # Use io.BytesIO to treat the decompressed bytes as a file
        return pd.read_csv(io.BytesIO(decompressed_data), **kwargs)
    except FileNotFoundError:
        # Only print warning if we aren't just checking for a reference file silently
        if 'source_packet_received' not in path:
            print(f"Warning: File not found at {path}. Proceeding with empty data.")
        return pd.DataFrame(columns=['timestamp', 'value'])
    except Exception as e:
        print(f"An error occurred reading {path}: {e}")
        return pd.DataFrame(columns=['timestamp', 'value'])


def decompress_brotli(input_path, output_path=None):
    """
    Decompresses a .br file using streaming to handle large files efficiently.
    """
    # Verify input exists
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # Determine output filename automatically if not provided
    if output_path is None:
        if input_path.endswith('.br'):
            output_path = input_path[:-3]  # Remove .br extension
        else:
            output_path = input_path + '.out'

    print(f"Decompressing '{input_path}' -> '{output_path}'...")

    try:
        # Open files in binary mode
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            # Create a streaming decompressor
            decompressor = brotli.Decompressor()
            
            # Read and decompress in chunks (e.g., 64KB)
            while True:
                chunk = f_in.read(65536)
                if not chunk:
                    break
                # Process the chunk
                decompressed_chunk = decompressor.process(chunk)
                f_out.write(decompressed_chunk)
            
            if hasattr(decompressor, 'finish'):
                 f_out.write(decompressor.finish())

        print("Success!")

    except brotli.error as e:
        print(f"Decompression error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_erasure_rates(base_path, config_sequence, overlay_file=None, overlay_label='Overlay Data', reuse_base_scale=False, save_path=None):
    """
    Plots erasure rates using absolute timestamps for phase alignment.
    Fix: Ensures last phase label centers within the visible plot limit.
    """
    
    # --- 1. Load Base Data ---
    df_channel = read_brotli_csv(os.path.join(base_path, 'controller/controller_local_erasure_rate.csv.br'))
    df_e2e = read_brotli_csv(os.path.join(base_path, 'controller/controller_e2e_erasure_rate.csv.br'))
    df_corrected = read_brotli_csv(os.path.join(base_path, 'controller/controller_corrected_erasure_rate.csv.br'))

    df_overlay = None
    if overlay_file:
        df_overlay = read_brotli_csv(os.path.join(base_path, overlay_file))

    # --- 2. Load Phase Timing Data (Absolute Timestamps) ---
    phase_timing_path = os.path.join(base_path, 'phase_switches.csv')
    df_phases = None
    
    if os.path.exists(phase_timing_path):
        try:
            # Rust writes raw numbers without a header, so we specify the column name manually
            df_phases = pd.read_csv(phase_timing_path, header=None, names=['timestamp'])
            print(f"Loaded precise phase timing from {phase_timing_path} ({len(df_phases)} phases)")
        except Exception as e:
            print(f"Warning: Could not read phase timing file: {e}")

    # --- 3. Normalize Timestamps (Unified) ---
    base_dfs = [df for df in [df_channel, df_e2e, df_corrected] if not df.empty]
    if not base_dfs:
        print("Error: All base dataframes are empty. Cannot generate plot.")
        return
        
    # Gather all timestamps to find the global start (t=0)
    timestamps = []
    for df in base_dfs:
        timestamps.extend([df['timestamp'].min(), df['timestamp'].max()])

    if df_overlay is not None and not df_overlay.empty:
        timestamps.extend([df_overlay['timestamp'].min(), df_overlay['timestamp'].max()])
    
    if df_phases is not None and not df_phases.empty:
        timestamps.extend([df_phases['timestamp'].min(), df_phases['timestamp'].max()])

    if not timestamps:
        print("Error: No timestamp data available.")
        return

    # --- NEW: Reference Time Extraction ---
    # Determine min_ts (t=0) from source_packet_received.csv.br if available
    ref_path = os.path.join(base_path, 'controller/source_packet_received.csv.br')
    min_ts = None
    
    try:
        # Read only the first row (nrows=1) to get the start time efficiently
        df_ref = read_brotli_csv(ref_path, nrows=1)
        if not df_ref.empty:
            # Extract col 0, val 0
            min_ts = df_ref.iloc[0, 0]
            print(f"Reference time (t=0) set to {min_ts} from {ref_path}")
    except Exception as e:
        print(f"Could not extract reference time: {e}")

    # Fallback if reference file logic failed
    if min_ts is None:
        min_ts = min(timestamps)
        print(f"Reference file missing. Fallback t=0 set to min timestamp: {min_ts}")

    max_ts = max(timestamps)
    total_duration = (max_ts - min_ts) / 1e9

    # Determine Plot Limit based on E2E data specifically
    plot_end_limit = total_duration # Default fallback
    if not df_e2e.empty:
        max_e2e_ts = df_e2e['timestamp'].max()
        plot_end_limit = (max_e2e_ts - min_ts) / 1e9 + 0.3  # Add 300ms buffer

    # Convert traces to seconds relative to global start
    for df in base_dfs:
        df['time_sec'] = (df['timestamp'] - min_ts) / 1e9
        
    if df_overlay is not None and not df_overlay.empty:
        df_overlay['time_sec'] = (df_overlay['timestamp'] - min_ts) / 1e9
        
    # Convert phases to seconds relative to global start
    if df_phases is not None and not df_phases.empty:
        df_phases['time_sec'] = (df_phases['timestamp'] - min_ts) / 1e9

    # --- 4. Prepare Log Data ---
    MIN_VAL = 1e-9
    def clean_log_data(series):
        return series.apply(lambda x: max(x, MIN_VAL))

    for df in base_dfs:
        df['value_plot'] = clean_log_data(df['value'])

    # --- 5. Create Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Lines
    if not df_channel.empty:
        ax.plot(df_channel['time_sec'], df_channel['value_plot'], 
                label='Channel Erasure Rate', color='tab:blue', alpha=0.8, zorder=2)
    
    if not df_corrected.empty:
        ax.plot(df_corrected['time_sec'], df_corrected['value_plot'], 
                label='Corrected Erasure Rate', color='purple', linestyle='-.', alpha=0.8, zorder=3)

    if not df_e2e.empty:
        ax.plot(df_e2e['time_sec'], df_e2e['value_plot'], 
                label='E2E Erasure Rate', color='#39FF14', linewidth=2.5, alpha=0.9, zorder=10)

    # Optional Overlay
    ax2 = None
    if df_overlay is not None and not df_overlay.empty:
        if reuse_base_scale:
            df_overlay['value_plot'] = clean_log_data(df_overlay['value'])
            ax.plot(df_overlay['time_sec'], df_overlay['value_plot'], 
                    label=f"{overlay_label} (Log)", color='black', linewidth=1.5, alpha=0.6, linestyle='--')
        else:
            ax2 = ax.twinx()
            ax2.plot(df_overlay['time_sec'], df_overlay['value'], 
                     label=overlay_label, color='black', linewidth=1, alpha=0.4)
            limit = df_overlay['value'].abs().max() * 1.1 if df_overlay['value'].abs().max() != 0 else 1.0
            ax2.set_ylim(-limit, limit)
            ax2.set_ylabel(f'{overlay_label} ', fontsize=14)

    # --- 6. Dynamic Phase Annotations ---
    total_packets = sum(c[4] for c in config_sequence)
    
    phase_boundaries = []
    
    if df_phases is not None and not df_phases.empty:
        # METHOD A: USE PRECISE LOGS (Now fully aligned via `time_sec`)
        times = df_phases.sort_values('timestamp')['time_sec'].values
        
        # filtering and zero-anchoring logic
        times = [t for t in times if t >= 0] 
        if len(times) > 0 and times[0] > 1.0: 
             times = np.insert(times, 0, 0.0)
        elif len(times) == 0:
             times = [0.0]
        
        for i, config in enumerate(config_sequence):
            if i < len(times):
                start = times[i]
                # Determine raw end based on next phase or total duration
                raw_end = times[i+1] if i + 1 < len(times) else total_duration
                
                end = min(raw_end, plot_end_limit)
                
                phase_boundaries.append((start, end, config))
    else:
        # METHOD B: FALLBACK ESTIMATION (If logs missing)
        if total_packets > 0:
            current_start = 0.0
            for i, config in enumerate(config_sequence):
                packet_count = config[4]
                phase_duration = (packet_count / total_packets) * total_duration
                current_end = current_start + phase_duration
                
                # FIX: Clamp the end to the plot limit for visualization
                vis_end = min(current_end, plot_end_limit)
                
                phase_boundaries.append((current_start, vis_end, config))
                current_start = current_end

    # Render Phase Annotations
    # Adjusted height using constant
    y_text_pos = 1.5 - PHASE_LABEL_VERTICAL_OFFSET
    
    for i, (start, end, config) in enumerate(phase_boundaries):
        loss_rate, correlation, rtt, target_loss, packet_count = config
        
        # Don't plot phases that start after the plot limit
        if start >= plot_end_limit:
            continue

        lbl_chan = 'True Avg Ch Erasure Rate' if i == 0 else "_"
        lbl_targ = 'Target Erasure Rate' if i == 0 else "_"

        ax.hlines(y=loss_rate, xmin=start, xmax=end, color='red', linestyle='--', linewidth=1.5, label=lbl_chan)
        ax.hlines(y=target_loss, xmin=start, xmax=end, color='darkgreen', linestyle=':', linewidth=1.5, label=lbl_targ)
        
        if i < len(phase_boundaries) - 1:
            ax.axvline(x=end, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        mid_point = start + (end - start) / 2
        loss_str = f"{loss_rate}" if loss_rate > 0 else "0"
        
        # Only label if phase is wide enough to be readable
        # AND if we haven't clamped it to be effectively invisible
        if (end - start) > (plot_end_limit * 0.05):
            ax.text(mid_point, y_text_pos, 
                    f'Phase {i+1}\n$p_e = {loss_str}$\n$\\rho = {correlation}$', 
                    ha='center', va='bottom', fontsize=13, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # --- 7. Formatting ---
    ax.set_yscale('log')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Loss Rate (Log)', fontsize=14)
    title_suffix = f" & {overlay_label}" if df_overlay is not None and not df_overlay.empty else ""
    ax.set_title(f'Packet Loss Rate{title_suffix}', fontsize=16)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    if config_sequence:
        min_target = min(c[3] for c in config_sequence)
        if min_target <= 0: min_target = MIN_VAL
        ax.set_ylim(bottom=min_target * 0.01, top=2.0)
    else:
        ax.set_ylim(bottom=MIN_VAL, top=2.0)

    # Apply X-Axis Limits
    ax.set_xlim(left=0.0, right=plot_end_limit)

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ([], []) if ax2 is None else ax2.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, lines1 + lines2))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', ncol=3, frameon=True, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_erasure_distribution(base_path, config_sequence, bins=50, save_path=None):
    """
    Plots the distribution of E2E erasure rates using a bar plot with non-uniform bin widths.
    Includes boxplot-style statistical indicators.
    The target rate is indicated by a red tick label instead of a vertical line.
    """
    # --- 1. Load Data ---
    try:
        # Ensure read_brotli_csv is available or imported
        df_e2e = read_brotli_csv(os.path.join(base_path, 'controller/controller_e2e_erasure_rate.csv.br'))
    except NameError:
        print("Error: 'read_brotli_csv' is not defined. Please ensure it is imported.")
        return

    if df_e2e.empty:
        print("Error: E2E dataframe is empty. Cannot generate plot.")
        return

    values = df_e2e['value'].dropna().values
    total_samples = len(values)

    # --- 2. Determine Target Erasure Rate (Center) ---
    targets = [c[3] for c in config_sequence if c[3] > 0]
    if not targets:
        print("Error: No valid target erasure rates found.")
        return
    target_erasure_rate = np.median(targets)

    # --- 3. Define Symmetric, Non-Uniform Bin Edges ---
    num_bins_per_side = bins // 2
    
    bin_edges_right = [target_erasure_rate]
    for i in range(1, num_bins_per_side + 1):
        bin_edges_right.append(target_erasure_rate * (1 + 0.1 * i))

    bin_edges_left = []
    for i in range(1, num_bins_per_side + 1):
        edge = target_erasure_rate * (1 - 0.1 * i)
        if edge > 0:
            bin_edges_left.append(edge)
        else:
            bin_edges_left.append(0)
            break
    bin_edges_left.sort()

    custom_bins = sorted(list(set(bin_edges_left + bin_edges_right)))

    data_min, data_max = values.min(), values.max()
    if data_min < custom_bins[0]:
        custom_bins.insert(0, data_min)
    if data_max > custom_bins[-1]:
        custom_bins.append(data_max)
    
    custom_bins = sorted(list(set(custom_bins)))
    bin_edges = np.array(custom_bins)
    
    # --- 4. Calculate Histogram ---
    counts, _ = np.histogram(values, bins=bin_edges)
    probabilities = counts / total_samples
    
    # --- 5. Calculate Boxplot Statistics ---
    q1 = np.percentile(values, 25)
    median = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    whisker_low = max(data_min, q1 - 1.5 * iqr)
    whisker_high = min(data_max, q3 + 1.5 * iqr)

    # --- 6. Helper: Map Data Value to Visual Axis ---
    def map_value_to_visual_x(value, edges):
        if value <= edges[0]: return -0.5
        if value >= edges[-1]: return len(edges) - 1.5 
        idx = np.searchsorted(edges, value) - 1
        bin_start = edges[idx]
        bin_end = edges[idx+1]
        fraction = (value - bin_start) / (bin_end - bin_start)
        return (idx - 0.5) + fraction

    vis_median = map_value_to_visual_x(median, bin_edges)
    vis_q1 = map_value_to_visual_x(q1, bin_edges)
    vis_q3 = map_value_to_visual_x(q3, bin_edges)
    vis_wh_low = map_value_to_visual_x(whisker_low, bin_edges)
    vis_wh_high = map_value_to_visual_x(whisker_high, bin_edges)

    # --- 7. Create Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))

    x_positions = np.arange(len(probabilities))
    ax.bar(x_positions, probabilities, color='#39FF14', edgecolor='black', alpha=0.6,
           width=1.0, align='center', label='E2E Erasure Samples')

    # Format Ticks
    tick_positions = np.arange(len(bin_edges)) - 0.5
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{edge:.1e}' for edge in bin_edges], rotation=90, ha='center', fontsize=8)

    # --- 8. Add Boxplot Lines ---
    ax.axvline(x=vis_median, color='blue', linestyle='-', linewidth=2.5, label=f'Median: {median:.1e}')
    ax.axvline(x=vis_q1, color='blue', linestyle='--', linewidth=1.5, label=f'IQR ({q1:.1e} - {q3:.1e})')
    ax.axvline(x=vis_q3, color='blue', linestyle='--', linewidth=1.5)
    ax.axvline(x=vis_wh_low, color='purple', linestyle=':', linewidth=2, label=f'Whiskers ({whisker_low:.1e}, {whisker_high:.1e})')
    ax.axvline(x=vis_wh_high, color='purple', linestyle=':', linewidth=2)

    # --- 9. Handle Target (Red Tick + Legend Only) ---
    # Find the index of the edge that matches the target rate
    closest_edge_idx = (np.abs(np.array(bin_edges) - target_erasure_rate)).argmin()
    
    # Get all tick labels (objects) so we can modify the specific one
    xtick_labels = ax.get_xticklabels()
    
    # Modify the specific tick label to be red and bold
    if 0 <= closest_edge_idx < len(xtick_labels):
        target_label = xtick_labels[closest_edge_idx]
        target_label.set_color('red')
        target_label.set_fontweight('bold')
        target_label.set_fontsize(10) # Slightly larger

    # Create a dummy legend handle for the target rate (since there's no line on the plot)
    target_handle = mlines.Line2D([], [], color='red', marker='|', linestyle='None',
                                  markersize=10, markeredgewidth=2, label=f'Target: {target_erasure_rate:.1e}')

    # --- 10. Final Formatting ---
    ax.set_xlabel('Erasure Rate Bins (Red Tick = Target)')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution')
    
    ax.grid(True, which="major", axis='x', linestyle=':', alpha=0.4, color='black')
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    
    # Combine existing handles with our custom target handle
    handles, labels = ax.get_legend_handles_labels()
    handles.append(target_handle)
    labels.append(target_handle.get_label())
    
    ax.legend(handles=handles, labels=labels, loc='upper right', framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
        
    plt.show()



def plot_redundancy_evolution(base_path, config_sequence, save_path=None):
    """
    Plots the cumulative ratio of Parity Packets / Source Packets (p/k) over time.
    Adds a 'Local' (Rolling) ratio that is highly smoothed based on the loss rate.
    Overlays the 'RI' (Redundancy Information) and 'Corrected Erasure Rate'.
    """
    # --- 1. Load Packet Data ---
    path_parity = os.path.join(base_path, 'sender/parity_packet_send.csv.br')
    path_source = os.path.join(base_path, 'sender/source_packet_send.csv.br')

    # Load only the first column (timestamp) using the generic loader
    df_p = read_brotli_csv(path_parity, header=None, usecols=[0], dtype=str)
    df_k = read_brotli_csv(path_source, header=None, usecols=[0], dtype=str)

    # Validate and standardize column names
    if not df_p.empty:
        df_p.columns = ['TS']
        df_p['TS'] = pd.to_numeric(df_p['TS'], errors='coerce')
        df_p.dropna(subset=['TS'], inplace=True)
        
    if not df_k.empty:
        df_k.columns = ['TS']
        df_k['TS'] = pd.to_numeric(df_k['TS'], errors='coerce')
        df_k.dropna(subset=['TS'], inplace=True)

    if df_p.empty or df_k.empty:
        print("Error: Packet dataframes are empty or could not be loaded.")
        return

    # --- 2. Process Packet Data ---
    df_p['type'] = 'p'
    df_k['type'] = 's'
    
    df_merged = pd.concat([df_p, df_k], ignore_index=True)
    df_merged.sort_values(by='TS', inplace=True)
    
    # Establish Global Start Time
    start_time = df_merged['TS'].iloc[0]
    
    # -- A. Global Cumulative Calculation --
    df_merged['is_parity'] = (df_merged['type'] == 'p').astype(int)
    df_merged['is_source'] = (df_merged['type'] == 's').astype(int)
    
    df_merged['cum_p'] = df_merged['is_parity'].cumsum()
    df_merged['cum_k'] = df_merged['is_source'].cumsum()
    
    valid_data = df_merged[df_merged['cum_k'] > 0].copy()
    
    MIN_VAL = 1e-9
    valid_data['ratio'] = np.maximum(valid_data['cum_p'] / valid_data['cum_k'], MIN_VAL)
    valid_data['rel_time'] = (valid_data['TS'] - start_time) / 1e9

    # --- 3. Phase Boundary Logic ---
    phase_timing_path = os.path.join(base_path, 'phase_switches.csv')
    df_phases = None
    if os.path.exists(phase_timing_path):
        try:
            df_phases = pd.read_csv(phase_timing_path, header=None, names=['timestamp'])
            df_phases['rel_time'] = (df_phases['timestamp'] - start_time) / 1e9
        except Exception as e:
            print(f"Warning: Could not read phase timing file: {e}")

    total_packets = sum(c[4] for c in config_sequence)
    phase_boundaries = []
    
    if df_phases is not None and not df_phases.empty:
        times = df_phases.sort_values('timestamp')['rel_time'].values
        times = [t for t in times if t >= 0] 
        if len(times) > 0 and times[0] > 1.0: 
             times = np.insert(times, 0, 0.0)
        elif len(times) == 0:
             times = [0.0]

        for i, config in enumerate(config_sequence):
            if i < len(times):
                start = times[i]
                end = times[i+1] if i + 1 < len(times) else valid_data['rel_time'].max()
                phase_boundaries.append((start, end, config))
    else:
        # Fallback Estimation
        if total_packets > 0:
            current_start = 0.0
            total_duration = valid_data['rel_time'].max()
            for i, config in enumerate(config_sequence):
                packet_count = config[4]
                phase_duration = (packet_count / total_packets) * total_duration
                current_end = current_start + phase_duration
                phase_boundaries.append((current_start, current_end, config))
                current_start = current_end

    # --- 4. Adaptive Local Calculation (Smoothed) ---
    valid_data['local_ratio'] = np.nan 

    for start, end, config in phase_boundaries:
        loss_rate, correlation, rtt, target_loss, _ = config
        
        effective_loss = loss_rate if loss_rate > 1e-5 else 1e-4
        adaptive_window = int(100 / effective_loss)
        
        if adaptive_window < 100:
             adaptive_window = 100

        mask = (valid_data['rel_time'] >= start) & (valid_data['rel_time'] <= end)
        if not mask.any():
            continue
            
        phase_subset = valid_data.loc[mask]
        rolling_prob = phase_subset['is_parity'].rolling(window=adaptive_window, min_periods=int(adaptive_window/2)).mean()
        
        denom = (1 - rolling_prob).clip(lower=MIN_VAL)
        local_r = np.maximum(rolling_prob / denom, MIN_VAL)
        
        valid_data.loc[mask, 'local_ratio'] = local_r

    valid_data['local_ratio'] = valid_data['local_ratio'].interpolate()

    # --- 5. Load External Metrics ---
    
    # A. Controller RI
    df_ri = read_brotli_csv(base_path + "controller/controller_schedule_update_ri.csv.br", 
                            header=None, usecols=[0, 1], dtype=str)
    
    if not df_ri.empty:
        df_ri.columns = ['TS', 'RI']
        df_ri['TS'] = pd.to_numeric(df_ri['TS'], errors='coerce')
        df_ri['RI'] = pd.to_numeric(df_ri['RI'], errors='coerce')
        df_ri.dropna(inplace=True)
        df_ri['rel_time'] = (df_ri['TS'] - start_time) / 1e9
        df_ri['RI'] = np.maximum(df_ri['RI'], MIN_VAL)
        df_ri = df_ri[df_ri['rel_time'] >= 0].copy()

    # B. Corrected Erasure Rate & PLOT LIMIT logic
    # Default limit: end of packet trace
    plot_end_limit = valid_data['rel_time'].max() 

    # Explicitly load E2E data to determine the axis limit
    e2e_path = os.path.join(base_path, 'controller/controller_e2e_erasure_rate.csv.br')
    df_e2e_limit = read_brotli_csv(e2e_path)
    
    if not df_e2e_limit.empty:
        max_ts_e2e = df_e2e_limit['timestamp'].max()
        # Calculate end time relative to THIS plot's start_time + 100ms
        plot_end_limit = (max_ts_e2e - start_time) / 1e9 + 0.1 

    # C. Corrected Erasure Rate (for plotting)
    df_corrected = read_brotli_csv(os.path.join(base_path, 'controller/controller_corrected_erasure_rate.csv.br'))
    if not df_corrected.empty and 'timestamp' in df_corrected.columns:
        df_corrected['rel_time'] = (df_corrected['timestamp'] - start_time) / 1e9
        df_corrected['value_plot'] = df_corrected['value'].apply(lambda x: max(x, MIN_VAL))
        df_corrected = df_corrected[df_corrected['rel_time'] >= 0].copy()

    # --- 6. Plotting ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot Local Ratio
    ax.plot(valid_data['rel_time'], valid_data['local_ratio'], 
            color='#17a2b8', linewidth=1.5, alpha=0.5, label='True Local RI', zorder=3)

    # Plot Global Ratio
    ax.plot(valid_data['rel_time'], valid_data['ratio'], 
            color='#007BFF', linewidth=2.0, label='Global True RI', zorder=5)
    
    # Plot Controller RI
    if not df_ri.empty:
        ax.plot(df_ri['rel_time'], df_ri['RI'], 
                color='#FF5733', linewidth=1.5, linestyle='-', label='Controller RI', zorder=6)

    # Plot Corrected Erasure Rate
    if not df_corrected.empty:
        ax.plot(df_corrected['rel_time'], df_corrected['value_plot'], 
                label='Corrected Erasure Rate', color='purple', linestyle='-.', alpha=0.8, zorder=4)

    # Render Phase Annotations & Lines
    for i, (start, end, config) in enumerate(phase_boundaries):
        loss_rate, correlation, _, target_loss, _ = config
        
        lbl_chan = 'True Avg Ch Erasure Rate' if i == 0 else "_"

        if loss_rate > 0:
            ax.hlines(y=loss_rate, xmin=start, xmax=end, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label=lbl_chan)
            
        # REMOVED: Target Erasure Rate line (hlines)
        
        if i < len(phase_boundaries) - 1:
            ax.axvline(x=end, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # --- 7. Formatting & Scaling ---
    ax.set_yscale('log')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Loss / Redundancy Ratio (Log)')
    ax.set_title('Evolution of Redundancy (Global vs Adaptive Local)')
    
    # Scale calculation
    ratio_low = valid_data['ratio'].quantile(0.001)
    ratio_high = valid_data['ratio'].quantile(0.999)
    
    if not df_ri.empty:
        ri_low = df_ri['RI'].quantile(0.001)
        ri_high = df_ri['RI'].quantile(0.999)
    else:
        ri_low = ratio_low
        ri_high = ratio_high

    global_min = min(ratio_low, ri_low)
    global_max = max(ratio_high, ri_high)
    global_min = max(global_min, MIN_VAL)
    
    if global_max <= global_min: 
        global_max = global_min * 10
    
    log_low = np.log10(global_min)
    log_high = np.log10(global_max)
    log_range = log_high - log_low
    margin_buffer = log_range * 0.4
    
    final_ymin = 10**(log_low - margin_buffer)
    final_ymax = 10**(log_high + margin_buffer)
    
    ax.set_ylim(bottom=final_ymin, top=final_ymax)
    
    # APPLY THE X-LIMIT (Ensures graph ends 300ms after last E2E measurement)
    ax.set_xlim(left=0.0, right=plot_end_limit)

    # --- Dynamic Text Positioning ---
    # Position text 5% down from the top edge of the visible log range
    y_text_val = 10**(np.log10(final_ymax) - (log_range * 0.05))

    for i, (start, end, config) in enumerate(phase_boundaries):
        loss_rate, correlation, _, _, _ = config
        loss_str = f"{loss_rate}" if loss_rate > 0 else "0"
        mid_point = start + (end - start) / 2
        
        if (end - start) > (plot_end_limit * 0.05):
             ax.text(mid_point, y_text_val, 
                    f'Phase {i+1}\n$p_e = {loss_str}$\n$\\rho = {correlation}$', 
                    ha='center', va='top', fontsize=9, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.grid(True, which="major", linestyle='-', alpha=0.5)
    ax.grid(True, which="minor", linestyle=':', alpha=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Redundancy evolution plot saved to {save_path}")
        
    plt.show()





def plot_subplots_erasure_rates_and_RI(data_dir, config_sequence, save_path=None):
    """
    Produces a single figure with two subplots sharing X axis:
      - Top: Erasure rates (based on plot_erasure_rates with Packet Debt overlay)
      - Bottom: Redundancy evolution (based on plot_redundancy_evolution)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.transforms as mtransforms
    import matplotlib.patches as mpatches
    import numpy as np
    import os


    MIN_VAL = 1e-9

    def clean_log_data(series):
        return series.apply(lambda x: max(x, MIN_VAL))

    # SHARED: Load phase timing
    phase_timing_path = os.path.join(data_dir, 'phase_switches.csv')
    df_phases_raw = None
    if os.path.exists(phase_timing_path):
        try:
            df_phases_raw = pd.read_csv(phase_timing_path, header=None, names=['timestamp'])
        except Exception as e:
            print(f"Warning: Could not read phase timing file: {e}")

    # PART A – Data loading for ERASURE RATES (top subplot)
    base_path = data_dir

    df_channel = read_brotli_csv(os.path.join(base_path, 'controller/controller_local_erasure_rate.csv.br'))
    df_e2e = read_brotli_csv(os.path.join(base_path, 'controller/controller_e2e_erasure_rate.csv.br'))
    df_corrected_top = read_brotli_csv(os.path.join(base_path, 'controller/controller_corrected_erasure_rate.csv.br'))
    overlay_label = 'Packet Debt'
    df_overlay = read_brotli_csv(os.path.join(base_path, 'controller/controller_schedule_update_packet_debt.csv.br'))

    base_dfs_top = [df for df in [df_channel, df_e2e, df_corrected_top] if not df.empty]
    if not base_dfs_top:
        print("Error: All base dataframes are empty for erasure rates subplot.")
        return

    # Determine reference time (t=0) from source_packet_received
    ref_path = os.path.join(base_path, 'controller/source_packet_received.csv.br')
    min_ts_top = None
    try:
        df_ref = read_brotli_csv(ref_path, nrows=1)
        if not df_ref.empty:
            min_ts_top = df_ref.iloc[0, 0]
    except Exception:
        pass

    timestamps_top = []
    for df in base_dfs_top:
        timestamps_top.extend([df['timestamp'].min(), df['timestamp'].max()])
    if df_overlay is not None and not df_overlay.empty:
        timestamps_top.extend([df_overlay['timestamp'].min(), df_overlay['timestamp'].max()])
    if df_phases_raw is not None and not df_phases_raw.empty:
        timestamps_top.extend([df_phases_raw['timestamp'].min(), df_phases_raw['timestamp'].max()])

    if min_ts_top is None:
        min_ts_top = min(timestamps_top)

    max_ts_top = max(timestamps_top)
    total_duration_top = (max_ts_top - min_ts_top) / 1e9

    # Plot end limit from E2E data
    plot_end_limit_top = total_duration_top
    if not df_e2e.empty:
        plot_end_limit_top = (df_e2e['timestamp'].max() - min_ts_top) / 1e9 + 0.3
    shared_xlim = plot_end_limit_top


    for df in base_dfs_top:
        df['time_sec'] = (df['timestamp'] - min_ts_top) / 1e9
        df['value_plot'] = clean_log_data(df['value'])

    if df_overlay is not None and not df_overlay.empty:
        df_overlay['time_sec'] = (df_overlay['timestamp'] - min_ts_top) / 1e9

    df_phases_top = None
    if df_phases_raw is not None and not df_phases_raw.empty:
        df_phases_top = df_phases_raw.copy()
        df_phases_top['time_sec'] = (df_phases_top['timestamp'] - min_ts_top) / 1e9

    # Phase boundaries for top subplot
    total_packets = sum(c[4] for c in config_sequence)
    phase_boundaries_top = []

    if df_phases_top is not None and not df_phases_top.empty:
        times = df_phases_top.sort_values('timestamp')['time_sec'].values
        for i, config in enumerate(config_sequence):
            if i < len(times):
                start = times[i]
                raw_end = times[i + 1] if i + 1 < len(times) else total_duration_top
                end = min(raw_end, plot_end_limit_top)
                phase_boundaries_top.append((start, end, config))
    else:
        if total_packets > 0:
            current_start = 0.0
            for i, config in enumerate(config_sequence):
                packet_count = config[4]
                phase_duration = (packet_count / total_packets) * total_duration_top
                current_end = current_start + phase_duration
                vis_end = min(current_end, plot_end_limit_top)
                phase_boundaries_top.append((current_start, vis_end, config))
                current_start = current_end

    # PART B – Data loading for REDUNDANCY EVOLUTION (bottom subplot)
    

    path_parity = os.path.join(base_path, 'sender/parity_packet_send.csv.br')
    path_source = os.path.join(base_path, 'sender/source_packet_send.csv.br')

    df_p = read_brotli_csv(path_parity, header=None, usecols=[0], dtype=str)
    df_k = read_brotli_csv(path_source, header=None, usecols=[0], dtype=str)

    if not df_p.empty:
        df_p.columns = ['TS']
        df_p['TS'] = pd.to_numeric(df_p['TS'], errors='coerce')
        df_p.dropna(subset=['TS'], inplace=True)
    if not df_k.empty:
        df_k.columns = ['TS']
        df_k['TS'] = pd.to_numeric(df_k['TS'], errors='coerce')
        df_k.dropna(subset=['TS'], inplace=True)

    if df_p.empty or df_k.empty:
        print("Error: Packet dataframes are empty for redundancy subplot.")
        return

    df_p['type'] = 'p'
    df_k['type'] = 's'
    df_merged = pd.concat([df_p, df_k], ignore_index=True)
    df_merged.sort_values(by='TS', inplace=True)

    start_time_bottom = df_merged['TS'].iloc[0]

    df_merged['is_parity'] = (df_merged['type'] == 'p').astype(int)
    df_merged['is_source'] = (df_merged['type'] == 's').astype(int)
    df_merged['cum_p'] = df_merged['is_parity'].cumsum()
    df_merged['cum_k'] = df_merged['is_source'].cumsum()

    valid_data = df_merged[df_merged['cum_k'] > 0].copy()
    valid_data['ratio'] = np.maximum(valid_data['cum_p'] / valid_data['cum_k'], MIN_VAL)
    valid_data['rel_time'] = (valid_data['TS'] - start_time_bottom) / 1e9

    # Phase boundaries for bottom subplot
    df_phases_bottom = None
    if df_phases_raw is not None and not df_phases_raw.empty:
        df_phases_bottom = df_phases_raw.copy()
        df_phases_bottom['rel_time'] = (df_phases_bottom['timestamp'] - start_time_bottom) / 1e9

    phase_boundaries_bottom = []
    if df_phases_bottom is not None and not df_phases_bottom.empty:
        times_b = df_phases_bottom.sort_values('timestamp')['rel_time'].values
        times_b = [t for t in times_b if t >= 0]
        if len(times_b) > 0 and times_b[0] > 1.0:
            times_b = np.insert(times_b, 0, 0.0)
        elif len(times_b) == 0:
            times_b = [0.0]
        for i, config in enumerate(config_sequence):
            if i < len(times_b):
                start = times_b[i]
                end = times_b[i + 1] if i + 1 < len(times_b) else valid_data['rel_time'].max()
                phase_boundaries_bottom.append((start, end, config))
    else:
        if total_packets > 0:
            current_start = 0.0
            total_dur_b = valid_data['rel_time'].max()
            for i, config in enumerate(config_sequence):
                packet_count = config[4]
                phase_duration = (packet_count / total_packets) * total_dur_b
                current_end = current_start + phase_duration
                phase_boundaries_bottom.append((current_start, current_end, config))
                current_start = current_end

    # Adaptive local ratio
    valid_data['local_ratio'] = np.nan
    for start, end, config in phase_boundaries_bottom:
        loss_rate, correlation, rtt, target_loss, _ = config
        effective_loss = loss_rate if loss_rate > 1e-5 else 1e-4
        adaptive_window = max(int(100 / effective_loss), 100)

        mask = (valid_data['rel_time'] >= start) & (valid_data['rel_time'] <= end)
        if not mask.any():
            continue
        phase_subset = valid_data.loc[mask]
        rolling_prob = phase_subset['is_parity'].rolling(window=adaptive_window, min_periods=int(adaptive_window / 2)).mean()
        denom = (1 - rolling_prob).clip(lower=MIN_VAL)
        local_r = np.maximum(rolling_prob / denom, MIN_VAL)
        valid_data.loc[mask, 'local_ratio'] = local_r

    valid_data['local_ratio'] = valid_data['local_ratio'].interpolate()

    # Controller RI
    df_ri = read_brotli_csv(data_dir + "controller/controller_schedule_update_ri.csv.br",
                            header=None, usecols=[0, 1], dtype=str)
    if not df_ri.empty:
        df_ri.columns = ['TS', 'RI']
        df_ri['TS'] = pd.to_numeric(df_ri['TS'], errors='coerce')
        df_ri['RI'] = pd.to_numeric(df_ri['RI'], errors='coerce')
        df_ri.dropna(inplace=True)
        df_ri['rel_time'] = (df_ri['TS'] - start_time_bottom) / 1e9
        df_ri['RI'] = np.maximum(df_ri['RI'], MIN_VAL)
        df_ri = df_ri[df_ri['rel_time'] >= 0].copy()


    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                          gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.15})
    #fig.suptitle("Erasure Rates with Packet Debt (a) and Redundancy Information (b)", fontsize=14, fontweight='bold', y=1.01)
    # TOP SUBPLOT

    # Lines
    if not df_channel.empty:
        ax_top.plot(df_channel['time_sec'], df_channel['value_plot'],
                    label='Channel Erasure Rate', color='tab:blue', alpha=0.8, zorder=2)
    if not df_corrected_top.empty:
        ax_top.plot(df_corrected_top['time_sec'], df_corrected_top['value_plot'],
                    label='Corrected Erasure Rate', color='purple', linestyle='-.', alpha=0.8, zorder=3)
    if not df_e2e.empty:
        ax_top.plot(df_e2e['time_sec'], df_e2e['value_plot'],
                    label='E2E Erasure Rate', color='#39FF14', linewidth=2.5, alpha=0.9, zorder=10)

    # Overlay (Packet Debt) on twin axis
    ax_top_twin = None
    if df_overlay is not None and not df_overlay.empty:
        ax_top_twin = ax_top.twinx()
        ax_top_twin.plot(df_overlay['time_sec'], df_overlay['value'],
                         label=overlay_label, color='black', linewidth=1, alpha=0.4)
        ax_top_twin.set_ylabel(f'{overlay_label} ', fontsize=14)
        # NOTE: set_ylim is intentionally deferred until after ax_top.set_ylim() is called below

    # Phase annotations – top
    y_text_pos_top = 0.25
    for i, (start, end, config) in enumerate(phase_boundaries_top):
        loss_rate, correlation, rtt, target_loss, packet_count = config
        if start >= shared_xlim:
            continue

        lbl_chan = 'True Avg Ch Erasure Rate' if i == 0 else "_"
        lbl_targ = 'Target Erasure Rate' if i == 0 else "_"

        ax_top.hlines(y=loss_rate, xmin=start, xmax=end, color='red', linestyle='--', linewidth=1.5, label=lbl_chan)
        ax_top.hlines(y=target_loss, xmin=start, xmax=end, color='darkgreen', linestyle=':', linewidth=1.5, label=lbl_targ)

        if i < len(phase_boundaries_top) - 1:
            ax_top.axvline(x=end, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        mid_point = start + (end - start) / 2
        loss_str = f"{loss_rate}" if loss_rate > 0 else "0"
        if (end - start) > (shared_xlim * 0.05):
            ax_top.text(mid_point, y_text_pos_top,
                        f'Phase {i + 1}\n$p_e = {loss_str}$\n$\\rho = {correlation}$',
                        ha='center', va='bottom', fontsize=13, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Formatting – top
    ax_top.set_yscale('log')
    ax_top.set_ylabel('Loss Rate (Log)', fontsize=14)
    title_suffix = f" & {overlay_label}"
    bbox = ax_top.get_window_extent()  # axes size in display (pixel) units
    ab_box_w = 0.03                     # desired width in axes coords
    ab_box_h = ab_box_w * (bbox.width / bbox.height)  # matching height in axes coords
    a_letter_heigth_mul_offset = 1.06
    ab_offset = mtransforms.ScaledTranslation(12/72, -12/72, fig.dpi_scale_trans)

    ax_top.add_patch(
        mpatches.Rectangle(
            (0.0, 1.0 - ab_box_h),
            ab_box_w, ab_box_h,
            transform=ax_top.transAxes + ab_offset,
            facecolor='white',
            edgecolor='black',
            linewidth=0.8
        )
    )
    ax_top.text(
        ab_box_w / 2, 1.0 - ab_box_h / 2 * a_letter_heigth_mul_offset, 'A',
        transform=ax_top.transAxes + ab_offset,
        ha='center', va='center',
        fontsize=16, fontweight='bold'
    )
    ax_top.set_title(f'Packet Loss Rate{title_suffix}', fontsize=16)
    ax_top.grid(True, which="both", ls="-", alpha=0.2)

    if config_sequence:
        min_target = min(c[3] for c in config_sequence)
        if min_target <= 0:
            min_target = MIN_VAL
        ax_top.set_ylim(bottom=min_target * 0.05, top=2.0)
    else:
        ax_top.set_ylim(bottom=MIN_VAL, top=2.0)

    # Align twin axis so that y=0 sits at the same height as min_target on the log scale
    if ax_top_twin is not None:
        ymin_log, ymax_log = ax_top.get_ylim()
        frac = (np.log10(min_target) - np.log10(ymin_log)) / (np.log10(ymax_log) - np.log10(ymin_log))
        frac = np.clip(frac, 0.01, 0.99)
        high = df_overlay['value'].abs().max() * 1.1 if df_overlay['value'].abs().max() != 0 else 1.0
        low = -frac / (1 - frac) * high
        ax_top_twin.set_ylim(low, high)

    # Legend – top (collected for shared legend below)
    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ([], []) if ax_top_twin is None else ax_top_twin.get_legend_handles_labels()
    by_label_top = dict(zip(labels1 + labels2, lines1 + lines2))

    # BOTTOM SUBPLOT

    ax_bot.plot(valid_data['rel_time'], valid_data['local_ratio'],
                color='#17a2b8', linewidth=1.5, alpha=0.5, label='True Local RI', zorder=3)
    #ax_bot.plot(valid_data['rel_time'], valid_data['ratio'],
    #            color='#007BFF', linewidth=2.0, label='Global True RI', zorder=5)

    if not df_ri.empty:
        ax_bot.plot(df_ri['rel_time'], df_ri['RI'],
                    color='#FF5733', linewidth=1.5, linestyle='-', label='Controller RI', zorder=6)

    

    # Phase annotations – bottom
    # Y-axis scaling for bottom
    ratio_low = valid_data['ratio'].quantile(0.001)
    ratio_high = valid_data['ratio'].quantile(0.999)
    if not df_ri.empty:
        ri_low = df_ri['RI'].quantile(0.001)
        ri_high = df_ri['RI'].quantile(0.999)
    else:
        ri_low, ri_high = ratio_low, ratio_high

    global_min = max(min(ratio_low, ri_low), MIN_VAL)
    global_max = max(ratio_high, ri_high)
    if global_max <= global_min:
        global_max = global_min * 10

    log_low = np.log10(global_min)
    log_high = np.log10(global_max)
    log_range = log_high - log_low
    margin_buffer = log_range * 0.4

    final_ymin = 10 ** (log_low - margin_buffer)
    final_ymax = 10 ** (log_high + margin_buffer)

    ax_bot.set_ylim(bottom=final_ymin, top=final_ymax)

    # y_text_val_bot = 10 ** (np.log10(final_ymax) - (log_range * 0.05))
    #y_text_val_bot = 1.5 - PHASE_LABEL_VERTICAL_OFFSET
    y_text_val_bot = 0.23

    for i, (start, end, config) in enumerate(phase_boundaries_bottom):
        loss_rate, correlation, _, target_loss, _ = config
        lbl_chan = 'True Avg Ch Erasure Rate' if i == 0 else "_"

        if loss_rate > 0:
            ax_bot.hlines(y=loss_rate, xmin=start, xmax=end, color='red', linestyle='--',
                          linewidth=2.5, alpha=0.8, label=lbl_chan)

        if i < len(phase_boundaries_bottom) - 1:
            ax_bot.axvline(x=end, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        mid_point = start + (end - start) / 2
        loss_str = f"{loss_rate}" if loss_rate > 0 else "0"
        if (end - start) > (shared_xlim * 0.05):
            ax_bot.text(mid_point, y_text_val_bot,
                        f'Phase {i + 1}\n$p_e = {loss_str}$\n$\\rho = {correlation}$',
                        ha='center', va='bottom', fontsize=13, color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Formatting – bottom
    ax_bot.set_yscale('log')
    ax_bot.set_xlabel('Time (seconds)', fontsize=14)
    ax_bot.set_ylabel('Loss / Redundancy Ratio (Log)', fontsize=14)
    #ax_bot.text(-0.02, 1.02, '(b)', transform=ax_bot.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='right')
    ax_bot.add_patch(
        mpatches.Rectangle(
            (0.0, 1.0 - ab_box_h),
            ab_box_w, ab_box_h,
            transform=ax_bot.transAxes + ab_offset,
            facecolor='white',
            edgecolor='black',
            linewidth=0.8
        )
    )
    b_letter_heigth_mul_offset = 1.1
    ax_bot.text(
        ab_box_w / 2, 1.0 - ab_box_h / 2 * b_letter_heigth_mul_offset, 'B',
        transform=ax_bot.transAxes + ab_offset,
        ha='center', va='center',
        fontsize=16, fontweight='bold'
    )
    ax_bot.set_title('Redundancy Evolution', fontsize=16)
    #ax_bot.grid(False)
    ax_bot.grid(True, which="both", ls="-", alpha=0.2)

    lines_bot, labels_bot = ax_bot.get_legend_handles_labels()
    by_label_bot = dict(zip(labels_bot, lines_bot))

    # SHARED X LIMIT
    ax_top.set_xlim(left=0.0, right=shared_xlim)

    # Leave vertical space at the bottom of the figure for the shared legend
    plt.tight_layout(rect=[0, 0.0, 1, 1])

    # Legend
    leg_top = ax_top.legend(
        by_label_top.values(), by_label_top.keys(),
        loc='lower right',
        ncol=3,
        frameon=True,
        fontsize=12,
    )

    leg_bot = ax_bot.legend(
        by_label_bot.values(), by_label_bot.keys(),
        loc='lower right',
        ncol=3,
        frameon=True,
        fontsize=12,
    )

    fig.add_artist(leg_top)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined subplot saved to {save_path}")

    plt.show()
