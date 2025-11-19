import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pathlib import Path

class SquidAngleGUI:
    def __init__(self, master):
        self.master = master
        master.title("Squid Attitude & Acoustic Incidence Angle Calculator V1.0 © Tong 2025")
        master.geometry("1000x750")

        # --- Main Frames ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Control Parameters", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        # Yaw
        ttk.Label(control_frame, text="Direction (Yaw ψ, 0-360°):").pack(pady=(0, 5))
        self.psi_slider = ttk.Scale(control_frame, from_=0, to=360, orient=tk.HORIZONTAL, length=200, command=lambda v: self.update_plot())
        self.psi_slider.set(45)
        self.psi_slider.pack(pady=(0, 10))
        self.psi_label = ttk.Label(control_frame, text=f"{self.psi_slider.get():.0f}°")
        self.psi_label.pack()

        # Tilt
        ttk.Label(control_frame, text="Tilt (Tilt τ, -90-90°):").pack(pady=(10, 5))
        self.tau_slider = ttk.Scale(control_frame, from_=-90, to=90, orient=tk.HORIZONTAL, length=200, command=lambda v: self.update_plot())
        self.tau_slider.set(20)
        self.tau_slider.pack(pady=(0, 10))
        self.tau_label = ttk.Label(control_frame, text=f"{self.tau_slider.get():.0f}°")
        self.tau_label.pack()

        # Z-Position
        ttk.Label(control_frame, text="Squid Z-axis Position:").pack(pady=(10, 5))
        self.z_pos_var = tk.StringVar(value="-0.5")
        self.z_pos_entry = ttk.Entry(control_frame, textvariable=self.z_pos_var, width=10)
        self.z_pos_entry.pack()

        # Update Button (still useful for Z-pos entry)
        self.update_button = ttk.Button(control_frame, text="Update Plot & Calculate", command=self.update_plot)
        self.update_button.pack(pady=20)

        # Results Display
        ttk.Label(control_frame, text="Calculation Results (Incidence Angles):").pack(pady=(10, 5))
        self.result_text = tk.Text(control_frame, height=10, width=35, wrap="word")
        self.result_text.pack()

        # --- Matplotlib Chart ---
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial plot
        self.update_plot()

    def update_plot(self, event=None):
        # Update slider labels
        self.psi_label.config(text=f"{self.psi_slider.get():.0f}°")
        self.tau_label.config(text=f"{self.tau_slider.get():.0f}°")
        
        self.ax.clear()

        psi_deg = self.psi_slider.get()
        tau_deg = self.tau_slider.get()
        try:
            squid_start_z = float(self.z_pos_var.get())
        except ValueError:
            squid_start_z = 0.0

        psi = np.deg2rad(psi_deg)
        tau = np.deg2rad(tau_deg)

        # 1. World coordinate system axes
        arrow_length = 1.5
        self.ax.quiver(0, 0, 0, arrow_length, 0, 0, color='gray', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, arrow_length, 0, color='gray', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, arrow_length, color='gray', arrow_length_ratio=0.1)

        text_offset = 0.1
        self.ax.text(arrow_length + text_offset, 0, 0, 'X', color='gray', fontsize='small')
        self.ax.text(0, arrow_length + text_offset, 0, 'Y', color='gray', fontsize='small')
        self.ax.text(0, 0, arrow_length + text_offset, 'Z', color='gray', fontsize='small')

        # 2. Beam vectors
        beam_vectors = {
            'North #1': np.array([0, 1, 0]), 'South #2': np.array([0, -1, 0]),
            'West #3': np.array([-1, 0, 0]), 'East #4': np.array([1, 0, 0]),
            'Vertical #5': np.array([0, 0, -1])
        }
        beam_colors = {'North #1': 'cyan', 'South #2': 'blue', 'West #3': 'green', 'East #4': 'lime', 'Vertical #5': 'magenta'}
        for name, vec in beam_vectors.items():
            self.ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=beam_colors[name], arrow_length_ratio=0.1, label=name)

        # 3. Squid attitude vector
        squid_vector = np.array([np.cos(tau) * np.cos(psi), np.cos(tau) * np.sin(psi), np.sin(tau)])
        self.ax.quiver(0, 0, squid_start_z, squid_vector[0], squid_vector[1], squid_vector[2],
                       color='red', linewidth=2.5, arrow_length_ratio=0.15, label=f'Squid (ψ={psi_deg:.0f}°, τ={tau_deg:.0f}°)')

        # 4. Calculate and display incidence angles
        self.result_text.delete(1.0, tk.END)
        result_str = f"Attitude: Yaw={psi_deg:.1f}°, Tilt={tau_deg:.1f}°\n" + "-"*30 + "\n"
        for name, vec in beam_vectors.items():
            dot_product = np.clip(np.dot(vec, squid_vector), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.rad2deg(angle_rad)
            result_str += f"{name:<12}: {angle_deg:.2f} degrees\n"
        self.result_text.insert(tk.END, result_str)

        # 5. Chart styling
        self.ax.set_xlim([-1.5, 1.5]); self.ax.set_ylim([-1.5, 1.5]); self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_xlabel('East-West Axis (X)', fontsize='small'); self.ax.set_ylabel('North-South Axis (Y)', fontsize='small'); self.ax.set_zlabel('Up-Down Axis (Z)', fontsize='small')
        self.ax.set_title('Echosounder Coordinate System & Squid Orientation', fontsize='medium')
        self.ax.legend(fontsize='small')
        self.ax.grid(True)
        self.ax.view_init(elev=25, azim=-50)

        self.canvas.draw()

if __name__ == '__main__':
    import os
    import sys

    def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    root = tk.Tk()
    try:
        root.iconbitmap(resource_path('icon.ico'))
    except Exception as e:
        print(f"Failed to load icon: {e}")
        
    app = SquidAngleGUI(root)
    root.mainloop()
