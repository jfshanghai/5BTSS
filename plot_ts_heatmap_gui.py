
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class TSHeatmapGUI:
    def __init__(self, master):
        self.master = master
        master.title("Target Strength (TS) Heatmap Viewer")
        master.geometry("1000x800")

        # --- File Paths ---
        self.LUT_FILE = 'incidence_angle_lut_fixed.nc'
        self.TS_FILE = 'TS_interpolated_0p1kHz.nc'
        
        # --- Data holders ---
        self.lut_da = None
        self.ts_ds = None
        self.ts_angle_range = None

        # --- Main Frames ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Control Parameters", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        ttk.Label(control_frame, text="Select Transducer:").pack(pady=(0, 5))
        self.transducer_var = tk.StringVar()
        self.transducer_combo = ttk.Combobox(control_frame, textvariable=self.transducer_var, state="readonly")
        self.transducer_combo.pack(pady=(0, 10))

        self.plot_button = ttk.Button(control_frame, text="Generate Heatmap", command=self.generate_heatmap)
        self.plot_button.pack(pady=10)
        
        self.status_var = tk.StringVar(value="Loading data...")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=180).pack(pady=(10,0))

        # --- Matplotlib Canvas ---
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.cbar = None # Placeholder for the colorbar

        # --- Load data and initialize ---
        self.load_data()
        if self.lut_da is not None:
             # Populate combobox after data is loaded
            beam_names = self.lut_da.coords['beam'].values.tolist()
            self.transducer_combo['values'] = beam_names
            self.transducer_combo.set(beam_names[0]) # Set default
            self.generate_heatmap() # Generate initial plot

    def load_data(self):
        if not os.path.exists(self.LUT_FILE) or not os.path.exists(self.TS_FILE):
            messagebox.showerror("Error", f"Could not find required data files:\n{self.LUT_FILE}\n{self.TS_FILE}")
            self.master.destroy()
            return
        
        try:
            lut_ds = xr.open_dataset(self.LUT_FILE)
            self.lut_da = lut_ds['__xarray_dataarray_variable__']
            self.ts_ds = xr.open_dataset(self.TS_FILE)
            
            min_angle = self.ts_ds.coords['angle'].min().item()
            max_angle = self.ts_ds.coords['angle'].max().item()
            self.ts_angle_range = (min_angle, max_angle)

            self.status_var.set("Data loaded. Select a transducer and click 'Generate'.")
        except Exception as e:
            messagebox.showerror("Data Loading Error", f"Failed to load or process data files.\nError: {e}")
            self.master.destroy()

    def generate_heatmap(self):
        if self.lut_da is None or self.ts_ds is None:
            self.status_var.set("Data not loaded. Cannot plot.")
            return

        selected_transducer = self.transducer_var.get()
        self.status_var.set(f"Processing for '{selected_transducer}'...")
        self.master.update_idletasks()

        # --- Parameters for calculation ---
        SQUID_ID = 'No1'
        FREQUENCY_KHZ = 120.0
        DEFAULT_TS = -110.0
        
        # Define resolution. Note: Higher resolution takes longer.
        YAW_RES = 5
        TILT_RES = 5
        YAW_RANGE = np.arange(0, 361, YAW_RES)
        TILT_RANGE = np.arange(-90, 91, TILT_RES)
        
        heatmap_data = np.zeros((len(TILT_RANGE), len(YAW_RANGE)))

        # --- Main Processing Loop ---
        min_ts_angle, max_ts_angle = self.ts_angle_range
        
        for i, tilt_deg in enumerate(TILT_RANGE):
            for j, yaw_deg in enumerate(YAW_RANGE):
                # Look up the specific incidence angle for this transducer
                # Apply 'nearest' method only to the numerical yaw and tilt coordinates.
                # 'beam' is selected via exact match in a separate .sel() call
                # to prevent errors if the beam coordinate is not monotonically sorted.
                incidence_angle = self.lut_da.sel(
                    yaw=yaw_deg, tilt=tilt_deg, method='nearest'
                ).sel(
                    beam=selected_transducer
                ).item()

                if min_ts_angle <= incidence_angle <= max_ts_angle:
                    ts_slice = self.ts_ds['TS'].sel(squid_id=SQUID_ID)
                    ts_val = ts_slice.sel(
                        frequency=FREQUENCY_KHZ, angle=incidence_angle, method='nearest'
                    ).item()
                else:
                    ts_val = DEFAULT_TS
                heatmap_data[i, j] = ts_val
        
        self.status_var.set("Plotting...")
        self.master.update_idletasks()

        # --- Plotting ---
        self.ax.clear()
        # If a colorbar already exists, remove its associated axes from the figure
        if self.cbar is not None:
            self.fig.delaxes(self.cbar.ax)
            self.cbar = None # Clear reference to the old colorbar

        im = self.ax.imshow(heatmap_data, extent=[0, 360, -90, 90], origin='lower', aspect='auto', cmap='jet')
        # Create a new colorbar for the new plot
        self.cbar = self.fig.colorbar(im, ax=self.ax)
        self.cbar.set_label('Target Strength (TS, dB)')

        self.ax.set_title(f'TS Heatmap for "{selected_transducer}" Transducer')
        self.ax.set_xlabel('Swimming Direction (Yaw, degrees)')
        self.ax.set_ylabel('Swimming Angle (Tilt, degrees)')
        
        self.canvas.draw()
        self.status_var.set("Heatmap generated successfully.")

    def on_closing(self):
        if self.ts_ds is not None:
            self.ts_ds.close()
        if self.lut_da is not None:
            self.lut_da.close()
        self.master.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = TSHeatmapGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
