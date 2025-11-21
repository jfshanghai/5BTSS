
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

def plot_ts_from_lut():
    """
    Calculates and plots the Target Strength (TS) for a squid at various tilt angles
    by looking up incidence angles from a pre-computed LUT.

    This script reads an incidence angle Look-Up Table (LUT) and a TS data file.
    For a fixed squid swimming direction (yaw=180°) and a range of swimming tilt
    angles (-60° to 20°), it looks up the corresponding incidence angles from the
    LUT. It then uses these angles to find the TS values at 120 kHz from the data
    file and plots the resulting curves.
    """
    # --- 1. Parameters ---
    SQUID_ID = 'No1'
    YAW_DEG = 180.0
    TILT_DEG_RANGE = np.arange(-60, 21, 1)
    FREQUENCY_KHZ = 120.0
    
    LUT_FILE = 'incidence_angle_lut_fixed.nc'
    TS_FILE = 'TS_interpolated_0p1kHz.nc'
    
    # Check if files exist
    if not os.path.exists(LUT_FILE) or not os.path.exists(TS_FILE):
        print(f"Error: One or both input files not found ('{LUT_FILE}', '{TS_FILE}')")
        return

    # --- 2. Load Datasets ---
    try:
        lut_ds = xr.open_dataset(LUT_FILE)
        # The LUT data is stored in the default variable when a DataArray is saved as a Dataset
        lut_da = lut_ds['__xarray_dataarray_variable__']
        ts_ds = xr.open_dataset(TS_FILE)
        print("All datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # --- 3. Look Up Incidence Angles and TS ---
    beam_names = lut_da.coords['beam'].values
    results = {name: [] for name in beam_names}
    
    for tilt_deg in TILT_DEG_RANGE:
        # Look up the 5 incidence angles for the given yaw and tilt
        incidence_angles = lut_da.sel(yaw=YAW_DEG, tilt=tilt_deg, method='nearest')
        
        for beam_name in beam_names:
            incidence_angle = incidence_angles.sel(beam=beam_name).item()
            
            # Look up TS value from the TS dataset
            # First, select the squid ID for an exact match
            ts_slice = ts_ds['TS'].sel(squid_id=SQUID_ID)
            # Then, use 'nearest' for both frequency and angle
            ts_val = ts_slice.sel(frequency=FREQUENCY_KHZ, angle=incidence_angle, method='nearest').item()
            
            results[beam_name].append(ts_val)

    lut_ds.close()
    ts_ds.close()
    print("TS lookup complete.")

    # --- 4. Plot Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for beam_name in beam_names:
        ax.plot(TILT_DEG_RANGE, results[beam_name], label=beam_name)

    ax.set_title(f'TS vs. Swimming Tilt for Squid "{SQUID_ID}" (Yaw={YAW_DEG}°, Freq={FREQUENCY_KHZ}kHz)')
    ax.set_xlabel('Squid Swimming Tilt Angle (τ, degrees)')
    ax.set_ylabel('Target Strength (TS, dB)')
    ax.legend(title='Transducer')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    output_filename = 'ts_from_lut_plot.png'
    # 修改：保存到 result/ 目录
    RESULT_DIR = "result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    plt.show()

if __name__ == '__main__':
    plot_ts_from_lut()
