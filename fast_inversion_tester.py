# -*- coding: utf-8 -*-
"""
快速反演测试脚本。

功能：
1.  直接加载已存在的统计查找表(LUT)，跳过耗时的生成过程。
2.  根据用户通过命令行参数设定的“真实”姿态，从LUT的统计数据中模拟一次观测。
3.  使用该模拟观测数据，运行贝叶斯反演。
4.  可视化反演结果，以快速评估算法在特定姿态下的表现。
"""

import numpy as np
import xarray as xr
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import os
import argparse  # 引入argparse模块

# --- 全局参数定义 (需要与生成LUT的脚本保持一致) ---
# 姿态空间定义 (1度分辨率)
YAW_ANGLES = np.arange(0, 360, 1)
TILT_ANGLES = np.arange(-60, 61, 1)
BEAM_DIFF_NAMES = ['N-V', 'S-V', 'W-V', 'E-V']

# 文件与目录名
STAT_LUT_FILENAME = "ts_diff_empirical_lut.nc" # 确保使用的是基于经验模型的LUT
RESULT_DIR = "result" # 新增：统一的结果输出目录

def simulate_observation_from_lut(true_yaw, true_tilt, stat_lut):
    """
    从已加载的LUT中，为给定的“真实”姿态模拟一次观测。
    """
    print(f"\n--- 正在从LUT为真实姿态 (Yaw={true_yaw}, Tilt={true_tilt}) 模拟观测数据 ---")
    
    try:
        closest_point = stat_lut.sel(yaw=true_yaw, tilt=true_tilt, method='nearest')
    except Exception as e:
        print(f"在LUT中查找姿态时出错: {e}")
        return None

    mu = closest_point['ts_diff_mean'].values
    sigma = closest_point['ts_diff_std'].values
    ts_diff_obs = [norm.rvs(loc=m, scale=s) for m, s in zip(mu, sigma)]
    
    print("TS差异观测值 (从LUT统计模型中采样):")
    for i, name in enumerate(BEAM_DIFF_NAMES):
        print(f"  - {name}: {ts_diff_obs[i]:.2f} dB")
        
    return ts_diff_obs

def run_bayesian_inversion(ts_diff_obs, stat_lut, yaw_prior_range=None, tilt_prior_range=None):
    """
    实现“贝叶斯解算器”，并允许引入先验知识。
    （此函数已从 `batch_inversion_analysis.py` 复制，支持先验）
    """
    print("\n--- 正在运行贝叶斯解算器 ---")
    start_time = time.time()
    
    # 计算似然度
    mu = stat_lut['ts_diff_mean'].values
    sigma = stat_lut['ts_diff_std'].values
    sigma = np.maximum(sigma, 1e-9)
    ts_obs_expanded = np.expand_dims(np.expand_dims(ts_diff_obs, axis=0), axis=0)
    
    log_likelihood_per_beam = -np.log(sigma) - (ts_obs_expanded - mu)**2 / (2 * sigma**2)
    total_log_likelihood = np.sum(log_likelihood_per_beam, axis=-1)

    # --- 构建先验概率分布 ---
    if yaw_prior_range is None and tilt_prior_range is None:
        # 如果未提供任何先验范围，则使用均匀先验
        print("    - 使用均匀先验 (无姿态偏好)")
        log_prior = np.zeros_like(total_log_likelihood)
    else:
        # 如果提供了先验范围，则构建一个非均匀先验
        print("    - 使用非均匀先验:")
        # 首先，假设所有姿态的对数先验概率都为-inf
        log_prior = np.full_like(total_log_likelihood, -1e9) # 用一个很大的负数代表log(0)

        # 创建一个覆盖所有姿态的布尔掩码
        prior_mask = np.ones_like(total_log_likelihood, dtype=bool)

        # 根据yaw_prior_range更新掩码
        if yaw_prior_range:
            print(f"      - Yaw 范围: {yaw_prior_range[0]}° to {yaw_prior_range[1]}°")
            yaw_coords_2d = np.broadcast_to(YAW_ANGLES[:, np.newaxis], prior_mask.shape)
            prior_mask &= (yaw_coords_2d >= yaw_prior_range[0]) & (yaw_coords_2d <= yaw_prior_range[1])

        # 根据tilt_prior_range更新掩码
        if tilt_prior_range:
            print(f"      - Tilt 范围: {tilt_prior_range[0]}° to {tilt_prior_range[1]}°")
            tilt_coords_2d = np.broadcast_to(TILT_ANGLES[np.newaxis, :], prior_mask.shape)
            prior_mask &= (tilt_coords_2d >= tilt_prior_range[0]) & (tilt_coords_2d <= tilt_prior_range[1])
        
        # 将满足所有先验条件的区域的对数先验概率设为0 (即概率为1)
        log_prior[prior_mask] = 0.0
    
    log_posterior = total_log_likelihood + log_prior
    
    # 防止数值下溢
    log_posterior -= np.max(log_posterior)
    posterior = np.exp(log_posterior)
    
    # 归一化
    posterior_sum = np.sum(posterior)
    if posterior_sum > 0:
        posterior /= posterior_sum
    else: # 如果所有后验概率都为0，避免除以零
        print("警告: 所有后验概率为零，可能先验设置过于严格导致无有效区域。")
        
    duration = time.time() - start_time
    print(f"--- 解算完成 (耗时: {duration:.2f} 秒) ---")
    return posterior

def visualize_posterior(posterior_map, true_yaw, true_tilt, yaw_coords, tilt_coords):
    """
    第四阶段：结果可视化。
    """
    print("\n--- 正在可视化后验概率分布 ---")
    
    max_prob_idx = np.unravel_index(np.argmax(posterior_map), posterior_map.shape)
    estimated_yaw = yaw_coords[max_prob_idx[0]]
    estimated_tilt = tilt_coords[max_prob_idx[1]]

    print(f"预设的真实姿态: Yaw={true_yaw}, Tilt={true_tilt}")
    print(f"估计的姿态:     Yaw={estimated_yaw}, Tilt={estimated_tilt} (后验概率最高点)")

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(posterior_map.T, origin='lower', aspect='auto',
                   extent=[yaw_coords[0], yaw_coords[-1], tilt_coords[0], tilt_coords[-1]],
                   cmap='viridis')

    ax.set_title('Posterior Probability Distribution (Fast Tester)')
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Tilt (degrees)')
    fig.colorbar(im, ax=ax, label='Probability')

    ax.plot(true_yaw, true_tilt, 'r+', markersize=15, markeredgewidth=2, label=f'True State ({true_yaw}, {true_tilt})')
    ax.plot(estimated_yaw, estimated_tilt, 'wx', markersize=15, markeredgewidth=2, label=f'Estimated State ({estimated_yaw}, {estimated_tilt})')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 修改：将结果保存到 result/ 文件夹
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = f"fast_test_result_yaw{true_yaw}_tilt{true_tilt}_prior.png" # 区分有无先验的图片
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"快速测试结果图已保存到 '{save_path}'")
    plt.show()

if __name__ == '__main__':
    # --- 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Fast inversion tester for squid orientation.')
    parser.add_argument('--yaw', type=float, default=180, help='True yaw angle to test (in degrees).')
    parser.add_argument('--tilt', type=float, default=-30, help='True tilt angle to test (in degrees).')
    parser.add_argument('--yaw-prior', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help='The prior range for yaw angle (e.g., --yaw-prior 10 170).')
    parser.add_argument('--tilt-prior', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help='The prior range for tilt angle (e.g., --tilt-prior 5 60).')
    
    args = parser.parse_args()

    # 使用命令行传入的参数
    TRUE_YAW = args.yaw
    TRUE_TILT = args.tilt
    # -----------------------------------------
    
    # 1. 加载LUT
    if not os.path.exists(STAT_LUT_FILENAME):
        print(f"错误: 查找表文件 '{STAT_LUT_FILENAME}' 不存在。")
        print("请先运行 `bayesian_solver.py` 来生成该文件。")
    else:
        print(f"成功加载查找表: '{STAT_LUT_FILENAME}'")
        stat_lut = xr.open_dataset(STAT_LUT_FILENAME)

        # 2. 从LUT模拟一次观测
        ts_observed = simulate_observation_from_lut(TRUE_YAW, TRUE_TILT, stat_lut)

        if ts_observed:
            # 3. 运行贝叶斯反演 (传入先验参数)
            posterior_distribution = run_bayesian_inversion(
                ts_observed, 
                stat_lut,
                yaw_prior_range=args.yaw_prior, 
                tilt_prior_range=args.tilt_prior
            )

            # 4. 可视化结果
            visualize_posterior(posterior_distribution, TRUE_YAW, TRUE_TILT, YAW_ANGLES, TILT_ANGLES)
