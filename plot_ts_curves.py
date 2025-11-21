
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

def plot_ts_vs_tilt():
    """
    Calculates and plots the Target Strength (TS) for a squid at various tilt angles.

    This script reads two TS data files, calculates the acoustic incidence angle for
    five transducers based on a fixed squid swimming direction (yaw=180°) and a
    range of swimming tilt angles (-60° to 20°). It then looks up the corresponding
    TS values at 120 kHz from the data files and plots the resulting curves.
    """
    # --- 1. Parameters ---
    SQUID_ID = 'No1'
    YAW_DEG = 180.0
    TILT_DEG_RANGE = np.arange(-60, 21, 1)
    FREQUENCY_KHZ = 120.0
    
    TS_FILE_1 = 'TS_interpolated_fixed.nc'
    TS_FILE_2 = 'TS_interpolated_0p1kHz.nc'
    
    # Check if files exist
    if not os.path.exists(TS_FILE_1) or not os.path.exists(TS_FILE_2):
        print(f"Error: One or both TS files not found ('{TS_FILE_1}', '{TS_FILE_2}')")
        return

    # --- 2. Load Datasets ---
    try:
        ts_ds1 = xr.open_dataset(TS_FILE_1)
        ts_ds2 = xr.open_dataset(TS_FILE_2)
        print("TS datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # --- 3. Define Beam Vectors ---
    beam_vectors = {
        'North': np.array([0, 1, 0]),
        'South': np.array([0, -1, 0]),
        'West': np.array([-1, 0, 0]),
        'East': np.array([1, 0, 0]),
        'Vertical': np.array([0, 0, -1])
    }
    
    # --- 4. Calculate Incidence Angles and Look Up TS ---
    results = {
        'file1': {name: [] for name in beam_vectors.keys()},
        'file2': {name: [] for name in beam_vectors.keys()}
    }
    
    psi = np.deg2rad(YAW_DEG)

    for tilt_deg in TILT_DEG_RANGE:
        tau = np.deg2rad(tilt_deg)
        
        # Calculate squid's body vector
        squid_vector = np.array([
            np.cos(tau) * np.cos(psi),
            np.cos(tau) * np.sin(psi),
            np.sin(tau)
        ])
        
        for beam_name, beam_vec in beam_vectors.items():
            # Calculate incidence angle based on the new definition
            dot_product = np.clip(np.dot(beam_vec, squid_vector), -1.0, 1.0)
            incidence_angle = np.rad2deg(np.arccos(dot_product)) - 90.0
            
            # First, select the squid ID for an exact match
            ts1_slice = ts_ds1['TS'].sel(squid_id=SQUID_ID)
            # Then, use 'nearest' for both frequency and angle on the remaining data
            ts_val1 = ts1_slice.sel(frequency=FREQUENCY_KHZ, angle=incidence_angle, method='nearest').item()

            # Repeat for the second dataset
            ts2_slice = ts_ds2['TS'].sel(squid_id=SQUID_ID)
            ts_val2 = ts2_slice.sel(frequency=FREQUENCY_KHZ, angle=incidence_angle, method='nearest').item()
            
            results['file1'][beam_name].append(ts_val1)
            results['file2'][beam_name].append(ts_val2)

    ts_ds1.close()
    ts_ds2.close()
    print("TS lookup complete.")

    # --- 5. Plot Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(beam_vectors)))
    
    for i, beam_name in enumerate(beam_vectors.keys()):
        # Plot from file 1 (solid line)
        ax.plot(TILT_DEG_RANGE, results['file1'][beam_name], 
                color=colors[i], 
                linestyle='-', 
                label=f'{beam_name} ({os.path.basename(TS_FILE_1)})')
        
        # Plot from file 2 (dashed line)
        ax.plot(TILT_DEG_RANGE, results['file2'][beam_name], 
                color=colors[i], 
                linestyle='--', 
                label=f'{beam_name} ({os.path.basename(TS_FILE_2)})')

    ax.set_title(f'TS vs. Swimming Tilt for Squid "{SQUID_ID}" (Yaw={YAW_DEG}°, Freq={FREQUENCY_KHZ}kHz)')
    ax.set_xlabel('Squid Swimming Tilt Angle (τ, degrees)')
    ax.set_ylabel('Target Strength (TS, dB)')
    ax.legend(title='Transducer & Source File', bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    
    output_filename = 'ts_vs_tilt_curves.png'
    # 修改：保存到 result/ 目录
    RESULT_DIR = "result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    plt.show()


if __name__ == '__main__':
    plot_ts_vs_tilt()
