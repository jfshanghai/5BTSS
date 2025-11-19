
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class TSPlotterGUI:
    def __init__(self, master):
        self.master = master
        master.title("Target Strength vs. Tilt Angle Plotter")
        master.geometry("1000x750")

        # --- File Paths ---
        self.LUT_FILE = 'incidence_angle_lut_fixed.nc'
        self.TS_FILE = 'TS_interpolated_0p1kHz.nc'

        # --- Main Frames ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Control Parameters", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        ttk.Label(control_frame, text="Swimming Direction (Yaw ψ, 0-360°):").pack(pady=(0, 5))
        self.yaw_var = tk.StringVar(value="180.0")
        self.yaw_entry = ttk.Entry(control_frame, textvariable=self.yaw_var, width=10)
        self.yaw_entry.pack(pady=(0, 10))

        self.plot_button = ttk.Button(control_frame, text="Generate Plot", command=self.update_plot)
        self.plot_button.pack(pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready. Press 'Generate Plot'.")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=180).pack(pady=(10,0))

        # --- Matplotlib Canvas ---
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- Pre-load data ---
        self.lut_da = None
        self.ts_ds = None
        self.ts_angle_range = None
        self.load_data()
        
        # Initial Plot
        self.update_plot()

    def load_data(self):
        self.status_var.set("Loading data...")
        if not os.path.exists(self.LUT_FILE) or not os.path.exists(self.TS_FILE):
            messagebox.showerror("Error", f"Could not find required data files:\n{self.LUT_FILE}\n{self.TS_FILE}")
            self.master.destroy()
            return
        
        try:
            lut_ds = xr.open_dataset(self.LUT_FILE)
            self.lut_da = lut_ds['__xarray_dataarray_variable__']
            self.ts_ds = xr.open_dataset(self.TS_FILE)
            
            # Get the valid angle range from the TS dataset
            min_angle = self.ts_ds.coords['angle'].min().item()
            max_angle = self.ts_ds.coords['angle'].max().item()
            self.ts_angle_range = (min_angle, max_angle)

            self.status_var.set(f"Data loaded. TS Angle Range: [{min_angle}°, {max_angle}°]")
        except Exception as e:
            messagebox.showerror("Data Loading Error", f"Failed to load or process data files.\nError: {e}")
            self.master.destroy()

    def update_plot(self):
        if self.lut_da is None or self.ts_ds is None:
            self.status_var.set("Data not loaded. Cannot plot.")
            return

        try:
            yaw_deg = float(self.yaw_var.get())
            if not (0 <= yaw_deg <= 360):
                raise ValueError("Yaw must be between 0 and 360.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid Yaw Angle: {e}")
            return
            
        self.status_var.set(f"Generating plot for Yaw = {yaw_deg}°...")
        self.master.update_idletasks() # Force GUI update

        # --- Parameters ---
        SQUID_ID = 'No1'
        TILT_DEG_RANGE = np.arange(-90, 91, 1) # Changed range to -90 to 90
        FREQUENCY_KHZ = 120.0
        DEFAULT_TS = -110.0 # Default TS value for missing angles
        
        # --- Processing ---
        beam_names = self.lut_da.coords['beam'].values
        results = {name: [] for name in beam_names} # Store list of TS values

        min_ts_angle, max_ts_angle = self.ts_angle_range

        for tilt_deg in TILT_DEG_RANGE:
            incidence_angles = self.lut_da.sel(yaw=yaw_deg, tilt=tilt_deg, method='nearest')
            
            for beam_name in beam_names:
                incidence_angle = incidence_angles.sel(beam=beam_name).item()
                
                # Check if the incidence angle is within the valid range of the TS data
                if np.all((incidence_angle >= min_ts_angle) & (incidence_angle <= max_ts_angle)):
                    ts_slice = self.ts_ds['TS'].sel(squid_id=SQUID_ID)
                    ts_val = ts_slice.sel(frequency=FREQUENCY_KHZ, angle=incidence_angle, method='nearest').item()
                else:
                    # Use the default value if no TS data is available for the angle
                    ts_val = DEFAULT_TS
                results[beam_name].append(ts_val)

        # --- Plotting ---
        self.ax.clear()
        
        for beam_name in beam_names:
            # Plot the continuous curve with default values for missing points
            self.ax.plot(TILT_DEG_RANGE, results[beam_name], label=beam_name, marker='.', markersize=2, linestyle='-')

        self.ax.set_title(f'TS vs. Swimming Tilt for Squid "{SQUID_ID}" (Yaw={yaw_deg}°, Freq={FREQUENCY_KHZ}kHz)')
        self.ax.set_xlabel('Squid Swimming Tilt Angle (τ, degrees)')
        self.ax.set_ylabel('Target Strength (TS, dB)')
        self.ax.legend(title='Transducer')
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        self.canvas.draw()
        self.status_var.set("Plot updated successfully.")

    def on_closing(self):
        # Clean up xarray datasets
        if self.ts_ds is not None:
            self.ts_ds.close()
        # The LUT dataset is part of self.lut_da
        if self.lut_da is not None:
            self.lut_da.close()
        self.master.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = TSPlotterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
