import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys

# --- File Paths ---
LUT_FILE = 'incidence_angle_lut_fixed.nc'
TS_FILE = 'TS_interpolated_0p1kHz.nc'

def load_data():
    """Loads the LUT and TS data files and returns the dataset objects."""
    if not os.path.exists(LUT_FILE) or not os.path.exists(TS_FILE):
        print(f"Error: Could not find required data files:\n{LUT_FILE}\n{TS_FILE}")
        return None, None

    try:
        lut_ds = xr.open_dataset(LUT_FILE)
        ts_ds = xr.open_dataset(TS_FILE)
        print("Data loaded successfully.")
        return lut_ds, ts_ds
    except Exception as e:
        print(f"Failed to load or process data files.\nError: {e}")
        return None, None

def generate_and_save_heatmap(lut_da, ts_ds, transducer_name):
    """Generates and saves a TS heatmap for a given transducer."""
    if lut_da is None or ts_ds is None:
        print("Data not loaded. Cannot plot.")
        return

    print(f"Processing for '{transducer_name}'...")

    # --- Parameters for calculation ---
    SQUID_ID = 'No1'
    FREQUENCY_KHZ = 120.0
    DEFAULT_TS = -110.0

    # Define resolution
    YAW_RES = 5
    TILT_RES = 5
    YAW_RANGE = np.arange(0, 361, YAW_RES)
    TILT_RANGE = np.arange(-90, 91, TILT_RES)

    heatmap_data = np.zeros((len(TILT_RANGE), len(YAW_RANGE)))

    # --- Main Processing Loop ---
    min_angle = ts_ds.coords['angle'].min().item()
    max_angle = ts_ds.coords['angle'].max().item()

    for i, tilt_deg in enumerate(TILT_RANGE):
        # Add a simple progress indicator
        sys.stdout.write(f"\r  Calculating... Tilt: {tilt_deg:>3}°")
        sys.stdout.flush()
        for j, yaw_deg in enumerate(YAW_RANGE):
            incidence_angle = lut_da.sel(
                yaw=yaw_deg, tilt=tilt_deg, method='nearest'
            ).sel(
                beam=transducer_name
            ).item()

            if min_angle <= incidence_angle <= max_angle:
                ts_slice = ts_ds['TS'].sel(squid_id=SQUID_ID)
                ts_val = ts_slice.sel(
                    frequency=FREQUENCY_KHZ, angle=incidence_angle, method='nearest'
                ).item()
            else:
                ts_val = DEFAULT_TS
            heatmap_data[i, j] = ts_val
    
    print("\nPlotting...")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # Using a consistent color range for better comparison across plots
    im = ax.imshow(heatmap_data, extent=[0, 360, -90, 90], origin='lower', aspect='auto', cmap='jet', vmin=-80, vmax=-30)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Target Strength (TS, dB)')

    ax.set_title(f'TS Heatmap for "{transducer_name}" Transducer')
    ax.set_xlabel('Swimming Direction (Yaw, degrees)')
    ax.set_ylabel('Swimming Angle (Tilt, degrees)')
    
    # Set explicit ticks
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save the figure
    # 修改：保存到 result/ 目录
    RESULT_DIR = "result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = f'ts_heatmap_{transducer_name}.png'
    save_path = os.path.join(RESULT_DIR, output_filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"Heatmap saved to {save_path}\n")


def main():
    """Main function to generate all heatmaps."""
    lut_ds, ts_ds = load_data()

    if lut_ds is None or ts_ds is None:
        return

    try:
        # The data variable name might be different, find it dynamically
        data_vars = list(lut_ds.data_vars)
        if not data_vars:
            print(f"Error: No data variables found in {LUT_FILE}")
            return
        lut_da = lut_ds[data_vars[0]]

        # Get the list of transducers (beams)
        transducer_names = lut_da.coords['beam'].values.tolist()
        print(f"Found transducers: {transducer_names}")

        for name in transducer_names:
            generate_and_save_heatmap(lut_da, ts_ds, name)
    
    finally:
        # Close datasets
        if ts_ds is not None:
            ts_ds.close()
        if lut_ds is not None:
            lut_ds.close()

    print("All heatmaps generated.")


if __name__ == '__main__':
    main()
