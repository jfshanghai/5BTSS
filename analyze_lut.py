# -*- coding: utf-8 -*-
"""
用于分析和可视化 `ts_diff_statistical_lut.nc` 文件内容的脚本。

功能：
1. 打印出 NetCDF 文件的数据结构。
2. 为4个TS差异特征分别绘制TS差异均值（ts_diff_mean）的热力图，以展示其在（Yaw, Tilt）姿态空间中的分布。
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 参数定义 ---
STAT_LUT_FILENAME = "ts_diff_empirical_lut.nc" # 更新为使用基于经验模型生成的LUT文件
OUTPUT_IMAGE_FILENAME = "lut_diff_heatmap_visualization.png"
RESULT_DIR = "result"

def analyze_and_visualize_lut():
    """
    主函数，执行文件分析和数据可视化。
    """
    # 检查文件是否存在
    if not os.path.exists(STAT_LUT_FILENAME):
        print(f"错误：未找到文件 '{STAT_LUT_FILENAME}'。")
        print("请先运行 `bayesian_solver.py` 以生成该文件 (在更新为TS差异模型后，需要重新生成)。")
        return

    # --- 1. 打印文件结构 ---
    print(f"--- 分析文件: {STAT_LUT_FILENAME} ---")
    
    # 使用 xarray 打开 NetCDF 文件
    with xr.open_dataset(STAT_LUT_FILENAME) as lut_ds:
        print("\n[ 文件结构与元数据 ]")
        print(lut_ds)
        
        # 提取坐标和数据
        yaw_coords = lut_ds['yaw'].values
        tilt_coords = lut_ds['tilt'].values
        beam_diff_names = lut_ds['beam_diff'].values
        ts_diff_mean_data = lut_ds['ts_diff_mean']

        # --- 2. 数据可视化 ---
        print("\n[ 正在生成TS差异数据分布热力图... ]")
        
        # 创建一个 2x2 的子图网格
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), constrained_layout=True)
        fig.suptitle('Mean TS Difference (vs Vertical) Distribution in LUT', fontsize=16)
        
        # 将2D的axes数组扁平化为1D，方便遍历
        ax_flat = axes.flatten()

        for i, beam_diff_name in enumerate(beam_diff_names):
            ax = ax_flat[i]
            
            # 选择当前特征的数据
            data_to_plot = ts_diff_mean_data.sel(beam_diff=beam_diff_name).T
            
            im = ax.imshow(data_to_plot, origin='lower', aspect='auto',
                           extent=[yaw_coords[0], yaw_coords[-1], tilt_coords[0], tilt_coords[-1]],
                           cmap='viridis')
            
            ax.set_title(f'Feature: {beam_diff_name}')
            ax.set_xlabel('Yaw (degrees)')
            ax.set_ylabel('Tilt (degrees)')
            
            # 添加颜色条
            fig.colorbar(im, ax=ax, label='Mean TS Difference (dB)')
            ax.grid(True, alpha=0.2)
            
        # 保存图像
        os.makedirs(RESULT_DIR, exist_ok=True)
        save_path = os.path.join(RESULT_DIR, OUTPUT_IMAGE_FILENAME)
        plt.savefig(save_path)
        print(f"可视化图像已保存到 '{save_path}'")
        
        # 显示图像
        plt.show()


if __name__ == '__main__':
    analyze_and_visualize_lut()