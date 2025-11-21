# -*- coding: utf-8 -*-
"""
用于根据真实的鱿鱼TS测量数据，构建和可视化经验TS模型。

功能：
1.  演示在单个频率下，如何通过对线性域的声学截面(σ)求平均，
    来构建一个物理意义正确的、平滑的经验TS插值模型。
2.  通过绘制二维热力图，可视化平均TS在整个频率和角度谱上的分布，
    以对比不同频率下的TS响应特性。
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# --- 参数定义 ---
DATA_FILENAME = "TS_interpolated_0p1kHz.nc"
TARGET_FREQ_KHZ = 120.0  # 用于单频模型可视化的目标频率
RESULT_DIR = "result" # 统一的结果输出目录

def visualize_single_frequency_model(freq_khz):
    """
    为指定的单个频率，构建并可视化经验TS模型。
    展示了从原始数据点到平滑插值曲线的完整过程。
    """
    print(f"\n--- 正在为 {freq_khz} kHz 构建和可视化单频TS模型 ---")
    
    # 1. 加载数据
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 '{DATA_FILENAME}' 未找到。")
        return
    ds = xr.open_dataset(DATA_FILENAME)

    # 2. 数据准备
    try:
        # 选择指定频率的数据
        ts_data_at_freq = ds['TS'].sel(frequency=freq_khz, method='nearest')
        actual_freq = ts_data_at_freq.frequency.item()
    except KeyError:
        print(f"错误: 在数据中找不到频率 {freq_khz} kHz。")
        return
        
    angles = ts_data_at_freq['angle'].values
    ts_squid1 = ts_data_at_freq.sel(squid_id='No1').values
    ts_squid2 = ts_data_at_freq.sel(squid_id='No2').values

    # 3. 在线性域(σ)中进行平均 (物理正确的方法)
    sigma_squid1 = 10**(ts_squid1 / 10)
    sigma_squid2 = 10**(ts_squid2 / 10)
    sigma_avg = (sigma_squid1 + sigma_squid2) / 2
    ts_avg = 10 * np.log10(sigma_avg)

    # 4. 构建插值模型
    interpolator = interp1d(angles, ts_avg, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # 5. 可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(angles, ts_squid1, 'o', markersize=4, alpha=0.5, label='Squid No1 (Original)')
    ax.plot(angles, ts_squid2, 's', markersize=4, alpha=0.5, label='Squid No2 (Original)')
    ax.plot(angles, ts_avg, 'D', markersize=5, color='red', label='Averaged TS (in linear domain)')

    smooth_angles = np.linspace(min(angles), max(angles), 200)
    smooth_ts = interpolator(smooth_angles)
    ax.plot(smooth_angles, smooth_ts, '-', linewidth=2, color='black', label='Interpolated TS Model')

    ax.set_title(f'Empirical TS Model Construction at {actual_freq:.1f} kHz')
    ax.set_xlabel('Incidence Angle (degrees)')
    ax.set_ylabel('Target Strength (TS, dB)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 修改：保存到 result/ 目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = f"ts_model_{actual_freq:.0f}kHz.png"
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"单频模型图已保存到 '{save_path}'")
    plt.show()

def visualize_full_frequency_spectrum():
    """
    可视化在整个“频率-角度”空间上的平均TS分布。
    """
    print("\n--- 正在可视化全频谱TS模型 ---")
    
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 '{DATA_FILENAME}' 未找到。")
        return
    ds = xr.open_dataset(DATA_FILENAME)
    
    ts_data = ds['TS'].values
    sigma_data = 10**(ts_data / 10)
    sigma_avg_2d = np.mean(sigma_data, axis=0)
    ts_avg_2d = 10 * np.log10(sigma_avg_2d)

    fig, ax = plt.subplots(figsize=(12, 8))
    angles = ds['angle'].values
    frequencies = ds['frequency'].values
    im = ax.pcolormesh(angles, frequencies, ts_avg_2d, cmap='viridis', shading='gouraud')
    ax.set_title('Mean TS across Full Frequency-Angle Spectrum')
    ax.set_xlabel('Incidence Angle (degrees)')
    ax.set_ylabel('Frequency (kHz)')
    fig.colorbar(im, ax=ax, label='Mean TS (dB)')
    
    # 修改：保存到 result/ 目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = "ts_model_full_spectrum.png"
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"全频谱模型图已保存到 '{save_path}'")
    plt.show()

def get_incidence_angles(yaw_deg, tilt_deg):
    """
    根据给定的姿态，计算5个波束的理想中心入射角。
    (此函数是从bayesian_solver.py复制而来，用于独立计算)
    """
    psi = np.deg2rad(yaw_deg)
    tau = np.deg2rad(tilt_deg)
    squid_vector = np.array([np.cos(tau) * np.cos(psi), np.cos(tau) * np.sin(psi), np.sin(tau)])
    beam_vectors = {'North': np.array([0, 1, 0]), 'South': np.array([0, -1, 0]), 'West': np.array([-1, 0, 0]), 'East': np.array([1, 0, 0]), 'Vertical': np.array([0, 0, -1])}
    
    incidence_angles = {}
    for name, b_vec in beam_vectors.items():
        dot_product = np.clip(np.dot(squid_vector, b_vec), -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(dot_product)) - 90.0
        incidence_angles[name] = angle_deg
    return incidence_angles

def compare_ts_differences_at_frequencies(freq_list_khz, tilt_for_comparison):
    """
    为指定的几个频率，对比“水平-垂直”TS差异随Yaw的变化曲线。
    """
    print(f"\n--- 正在对比 {freq_list_khz} kHz 在 Tilt={tilt_for_comparison}° 时的TS差异 ---")
    
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 '{DATA_FILENAME}' 未找到。")
        return
    ds = xr.open_dataset(DATA_FILENAME)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'TS Difference (Horiz - Vert) vs. Yaw at Tilt={tilt_for_comparison}°', fontsize=16)
    ax_flat = axes.flatten()
    beam_diff_names = ['N-V', 'S-V', 'W-V', 'E-V']
    
    for freq in freq_list_khz:
        ts_data_at_freq = ds['TS'].sel(frequency=freq, method='nearest')
        sigma_avg = np.mean(10**(ts_data_at_freq.values / 10), axis=0)
        ts_avg = 10 * np.log10(sigma_avg)
        interpolator = interp1d(ts_data_at_freq['angle'].values, ts_avg, kind='linear', bounds_error=False, fill_value='extrapolate')

        yaw_angles = np.arange(0, 361, 1)
        ts_diff_curves = {name: [] for name in beam_diff_names}

        for yaw in yaw_angles:
            inc_angles = get_incidence_angles(yaw, tilt_for_comparison)
            abs_ts = {name: interpolator(angle) for name, angle in inc_angles.items()}
            ts_v = abs_ts['Vertical']
            ts_diff_curves['N-V'].append(abs_ts['North'] - ts_v)
            ts_diff_curves['S-V'].append(abs_ts['South'] - ts_v)
            ts_diff_curves['W-V'].append(abs_ts['West'] - ts_v)
            ts_diff_curves['E-V'].append(abs_ts['East'] - ts_v)
        
        for i, name in enumerate(beam_diff_names):
            ax_flat[i].plot(yaw_angles, ts_diff_curves[name], label=f'{freq} kHz')

    for i, name in enumerate(beam_diff_names):
        ax_flat[i].set_title(f'Feature: {name}')
        ax_flat[i].set_xlabel('Yaw (degrees)')
        ax_flat[i].set_ylabel('TS Difference (dB)')
        ax_flat[i].grid(True, linestyle='--', alpha=0.6)
        ax_flat[i].legend()
        ax_flat[i].set_xticks(np.arange(0, 361, 45))

    # 修改：保存到 result/ 目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = "ts_diff_comparison.png"
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"TS差异对比图已保存到 '{save_path}'")
    plt.show()


if __name__ == '__main__':
    # 执行第一个任务：构建并可视化单频模型
    visualize_single_frequency_model(TARGET_FREQ_KHZ)
    
    # 执行第二个任务：可视化全频谱TS模型
    visualize_full_frequency_spectrum()
    
    # 执行第三个任务：对比70kHz和120kHz下的TS差异曲线
    compare_ts_differences_at_frequencies(freq_list_khz=[70.0, 120.0], tilt_for_comparison=0)
