# -*- coding: utf-8 -*-
"""
批量反演分析脚本。

功能：
1.  针对用户设定的“真实”姿态，重复执行N次独立的贝叶斯反演模拟。
2.  将每次模拟的真实姿态和预测姿态结果保存到CSV文件中。
3.  在所有模拟完成后，对结果进行统计分析，计算误差分布。
4.  生成可视化图表，展示预测结果的分布和误差的直方图。
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm # 引入tqdm来显示进度条

# --- 全局参数定义 (需要与生成LUT的脚本保持一致) ---
# 姿态空间定义 (1度分辨率)
YAW_ANGLES = np.arange(0, 360, 1)
TILT_ANGLES = np.arange(-60, 61, 1)
BEAM_DIFF_NAMES = ['N-V', 'S-V', 'W-V', 'E-V']

# 文件与目录名
STAT_LUT_FILENAME = "ts_diff_empirical_lut.nc"
RESULT_DIR = "result"

def simulate_observation_from_lut(true_yaw, true_tilt, stat_lut):
    """
    从已加载的LUT中，为给定的“真实”姿态模拟一次观测。
    （代码与 fast_inversion_tester.py 相同）
    """
    try:
        closest_point = stat_lut.sel(yaw=true_yaw, tilt=true_tilt, method='nearest')
    except Exception as e:
        print(f"在LUT中查找姿态时出错: {e}")
        return None

    mu = closest_point['ts_diff_mean'].values
    sigma = closest_point['ts_diff_std'].values
    ts_diff_obs = [norm.rvs(loc=m, scale=s) for m, s in zip(mu, sigma)]
        
    return ts_diff_obs

def run_bayesian_inversion(ts_diff_obs, stat_lut, yaw_prior_range=None, tilt_prior_range=None):
    """
    实现“贝叶斯解算器”，并允许引入先验知识。
    """
    # 计算似然度 (与之前版本相同)
    mu = stat_lut['ts_diff_mean'].values
    sigma = stat_lut['ts_diff_std'].values
    sigma = np.maximum(sigma, 1e-9)
    ts_obs_expanded = np.expand_dims(np.expand_dims(ts_diff_obs, axis=0), axis=0)
    
    log_likelihood_per_beam = -np.log(sigma) - (ts_obs_expanded - mu)**2 / (2 * sigma**2)
    total_log_likelihood = np.sum(log_likelihood_per_beam, axis=-1)

    # --- 新增: 构建先验概率分布 ---
    if yaw_prior_range is None and tilt_prior_range is None:
        # 如果未提供任何先验范围，则使用均匀先验
        print("    - 使用均匀先验 (无姿态偏好)")
        log_prior = np.zeros_like(total_log_likelihood)
    else:
        # 如果提供了先验范围，则构建一个非均匀先验
        print("    - 使用非均匀先验:")
        # 首先，假设所有姿态的对数先验概率都为-inf
        log_prior = np.full_like(total_log_likelihood, -1e9) # 用一个很大的负数代表log(0)

        # 创建一个覆盖所有姿态的布尔掩码，初始为True
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
    
    return posterior

def analyze_and_plot_results(df, true_yaw, true_tilt):
    """
    对DataFrame中的批量运行结果进行统计分析和可视化。
    """
    print("\n--- 正在对100次模拟结果进行统计分析 ---")
    
    # 计算误差，特别处理Yaw的循环特性 (-180, 180)
    yaw_error = (df['predicted_yaw'] - df['true_yaw'] + 180) % 360 - 180
    tilt_error = df['predicted_tilt'] - df['true_tilt']
    
    df['yaw_error'] = yaw_error
    df['tilt_error'] = tilt_error

    print("\n预测误差统计摘要:")
    print(df[['yaw_error', 'tilt_error']].describe())

    # --- 可视化 ---
    # 1. 预测结果散点图
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.scatter(df['predicted_yaw'], df['predicted_tilt'], alpha=0.6, label='Predicted States')
    ax1.plot(true_yaw, true_tilt, 'r+', markersize=20, markeredgewidth=3, label=f'True State ({true_yaw}, {true_tilt})')
    ax1.set_xlabel('Yaw (degrees)')
    ax1.set_ylabel('Tilt (degrees)')
    ax1.set_title(f'Distribution of {len(df)} Predictions')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks(np.arange(0, 361, 45))
    
    scatter_path = os.path.join(RESULT_DIR, f'batch_scatter_yaw{true_yaw}_tilt{true_tilt}.png')
    plt.savefig(scatter_path)
    print(f"\n预测分布散点图已保存到: '{scatter_path}'")

    # 2. 误差直方图
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))
    ax2.hist(df['yaw_error'], bins=20, edgecolor='black')
    ax2.set_title('Yaw Prediction Error Distribution')
    ax2.set_xlabel('Yaw Error (degrees)')
    ax2.set_ylabel('Frequency')
    
    ax3.hist(df['tilt_error'], bins=20, edgecolor='black')
    ax3.set_title('Tilt Prediction Error Distribution')
    ax3.set_xlabel('Tilt Error (degrees)')
    ax3.set_ylabel('Frequency')
    
    hist_path = os.path.join(RESULT_DIR, f'batch_hist_yaw{true_yaw}_tilt{true_tilt}.png')
    plt.savefig(hist_path)
    print(f"误差分布直方图已保存到: '{hist_path}'")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch inversion analysis for squid orientation.')
    parser.add_argument('--yaw', type=float, default=45, help='True yaw angle to test (in degrees).')
    parser.add_argument('--tilt', type=float, default=-20, help='True tilt angle to test (in degrees).')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of simulation runs.')
    parser.add_argument('--csv-path', type=str, default='batch_results.csv', help='Path to save the output CSV file.')
    parser.add_argument('--yaw-prior', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help='The prior range for yaw angle (e.g., --yaw-prior 10 170).')
    parser.add_argument('--tilt-prior', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help='The prior range for tilt angle (e.g., --tilt-prior 5 60).')

    args = parser.parse_args()

    TRUE_YAW = args.yaw
    TRUE_TILT = args.tilt
    NUM_RUNS = args.num_runs
    CSV_PATH = args.csv_path

    # 1. 加载LUT
    if not os.path.exists(STAT_LUT_FILENAME):
        print(f"错误: 查找表文件 '{STAT_LUT_FILENAME}' 不存在。")
        print("请先运行 `bayesian_solver.py` 来生成该文件。")
    else:
        print(f"成功加载查找表: '{STAT_LUT_FILENAME}'")
        stat_lut = xr.open_dataset(STAT_LUT_FILENAME)
        
        results_list = []
        print(f"\n--- 开始执行 {NUM_RUNS} 次模拟, 真实姿态 Yaw={TRUE_YAW}, Tilt={TRUE_TILT} ---")
        
        # 2. 循环执行模拟和反演
        # 用tqdm包装循环以显示进度条，并在循环描述中加入先验信息
        desc = "模拟进度"
        if args.yaw_prior or args.tilt_prior:
            desc += " (使用先验)"
        
        for _ in tqdm(range(NUM_RUNS), desc=desc):
            # a. 模拟一次观测
            ts_observed = simulate_observation_from_lut(TRUE_YAW, TRUE_TILT, stat_lut)
            if not ts_observed:
                print("模拟观测失败，跳过此次运行。")
                continue
            
            # b. 运行贝叶斯反演 (传入先验参数)
            posterior = run_bayesian_inversion(
                ts_observed, 
                stat_lut, 
                yaw_prior_range=args.yaw_prior, 
                tilt_prior_range=args.tilt_prior
            )
            
            # c. 找到预测结果
            max_prob_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
            pred_yaw = YAW_ANGLES[max_prob_idx[0]]
            pred_tilt = TILT_ANGLES[max_prob_idx[1]]
            
            # d. 记录结果
            results_list.append({
                'true_yaw': TRUE_YAW,
                'true_tilt': TRUE_TILT,
                'predicted_yaw': pred_yaw,
                'predicted_tilt': pred_tilt
            })
            
        # 3. 保存到CSV文件
        results_df = pd.DataFrame(results_list)
        # 确保目录存在
        output_dir = os.path.dirname(CSV_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(CSV_PATH, index=False)
        print(f"\n--- 模拟完成 ---")
        print(f"全部 {len(results_df)} 次运行结果已保存到 '{CSV_PATH}'")
        
        # 4. 分析和可视化
        if not results_df.empty:
            analyze_and_plot_results(results_df, TRUE_YAW, TRUE_TILT)
