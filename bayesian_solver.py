"""
使用贝叶斯方法从五维波束TS数据反演鱿鱼游泳姿态。

遵循《五维波束游泳状态解算.md》中描述的方法三。
此最终版本使用基于真实测量数据构建的经验TS模型。
"""

import numpy as np
import xarray as xr
from scipy.stats import norm
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
import os
import argparse # 引入argparse模块

# --- 全局参数定义 ---
# 姿态空间定义 (1度分辨率)
YAW_ANGLES = np.arange(0, 360, 1)
TILT_ANGLES = np.arange(-60, 61, 1)
BEAM_NAMES = ['North', 'South', 'West', 'East', 'Vertical']
BEAM_DIFF_NAMES = ['N-V', 'S-V', 'W-V', 'E-V']

# 模拟参数
MONTE_CARLO_SAMPLES = 1000
BEAM_UNCERTAINTY_STD = 3.5
TS_NOISE_STD = 0.5
MIN_L_CM = 17.0
MAX_L_CM = 20.0
REF_L_CM = 18.0

# 文件与目录名
REAL_DATA_FILENAME = "TS_interpolated_0p1kHz.nc"
STAT_LUT_FILENAME = "ts_diff_empirical_lut.nc" # 基于经验模型的新LUT
TARGET_FREQ_KHZ = 120.0 # 选择120kHz作为我们的模型频率
RESULT_DIR = "result" # 新增：统一的结果输出目录


class EmpiricalTSModel:
    """
    一个基于真实测量数据构建的经验TS模型类。
    """
    def __init__(self, data_filename, freq_khz):
        print(f"--- 正在为 {freq_khz} kHz 构建经验TS模型 ---")
        self.freq_khz = freq_khz
        self.interpolator = self._build_interpolator(data_filename, freq_khz)

    def _build_interpolator(self, data_filename, freq_khz):
        if not os.path.exists(data_filename):
            raise FileNotFoundError(f"错误: 经验模型所需的数据文件 '{data_filename}' 未找到。")
        
        with xr.open_dataset(data_filename) as ds:
            ts_data_at_freq = ds['TS'].sel(frequency=freq_khz, method='nearest')
            angles = ts_data_at_freq['angle'].values
            
            # 以物理正确的方式平均TS
            sigma_avg = np.mean(10**(ts_data_at_freq.values / 10), axis=0)
            ts_avg = 10 * np.log10(sigma_avg)

            # 创建插值函数
            print(f"成功构建基于 {ts_data_at_freq.frequency.item():.1f} kHz 真实数据的TS插值模型。")
            return interp1d(angles, ts_avg, kind='linear', bounds_error=False, fill_value='extrapolate')

    def get_ts(self, incidence_angle_deg):
        """ 使用插值器获取TS值 """
        return self.interpolator(incidence_angle_deg)


def get_incidence_angles(yaw_deg, tilt_deg):
    """ 根据给定的姿态，计算5个波束的理想中心入射角。 """
    psi, tau = np.deg2rad(yaw_deg), np.deg2rad(tilt_deg)
    squid_vector = np.array([np.cos(tau) * np.cos(psi), np.cos(tau) * np.sin(psi), np.sin(tau)])
    beam_vectors = {'North': [0, 1, 0], 'South': [0, -1, 0], 'West': [-1, 0, 0], 'East': [1, 0, 0], 'Vertical': [0, 0, -1]}
    
    incidence_angles = {}
    for name, b_vec in beam_vectors.items():
        dot_product = np.clip(np.dot(squid_vector, b_vec), -1.0, 1.0)
        incidence_angles[name] = np.rad2deg(np.arccos(dot_product)) - 90.0
    return incidence_angles


def generate_statistical_lut(empirical_model):
    """
    第一阶段：使用经验TS模型，构建包含TS差异均值和标准差的查找表(LUT)。
    """
    print("\n--- 阶段一：开始生成基于经验模型的统计查找表 (LUT) ---")
    start_time = time.time()
    
    ts_diff_means = np.zeros((len(YAW_ANGLES), len(TILT_ANGLES), len(BEAM_DIFF_NAMES)))
    ts_diff_stds = np.zeros((len(YAW_ANGLES), len(TILT_ANGLES), len(BEAM_DIFF_NAMES)))

    for i, yaw in enumerate(YAW_ANGLES):
        for j, tilt in enumerate(TILT_ANGLES):
            ideal_angles = get_incidence_angles(yaw, tilt)
            mc_ts_diff_samples = np.zeros((MONTE_CARLO_SAMPLES, len(BEAM_DIFF_NAMES)))

            for s_idx in range(MONTE_CARLO_SAMPLES):
                random_l = np.random.uniform(MIN_L_CM, MAX_L_CM)
                absolute_ts = {}
                for beam_name in BEAM_NAMES:
                    angle_with_uncertainty = ideal_angles[beam_name] + norm.rvs(scale=BEAM_UNCERTAINTY_STD)
                    
                    # 核心修改：使用经验模型
                    base_ts = empirical_model.get_ts(angle_with_uncertainty)
                    length_effect = 20 * np.log10(random_l / REF_L_CM)
                    ts_val = base_ts + length_effect
                    
                    final_ts = ts_val + norm.rvs(scale=TS_NOISE_STD)
                    absolute_ts[beam_name] = final_ts
                
                ts_v = absolute_ts['Vertical']
                mc_ts_diff_samples[s_idx, 0] = absolute_ts['North'] - ts_v
                mc_ts_diff_samples[s_idx, 1] = absolute_ts['South'] - ts_v
                mc_ts_diff_samples[s_idx, 2] = absolute_ts['West'] - ts_v
                mc_ts_diff_samples[s_idx, 3] = absolute_ts['East'] - ts_v

            ts_diff_means[i, j, :] = np.mean(mc_ts_diff_samples, axis=0)
            ts_diff_stds[i, j, :] = np.std(mc_ts_diff_samples, axis=0)
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {i+1}/{len(YAW_ANGLES)} 个Yaw角度...")

    lut = xr.Dataset(
        {"ts_diff_mean": (("yaw", "tilt", "beam_diff"), ts_diff_means), "ts_diff_std": (("yaw", "tilt", "beam_diff"), ts_diff_stds)},
        coords={ "yaw": YAW_ANGLES, "tilt": TILT_ANGLES, "beam_diff": BEAM_DIFF_NAMES }
    )
    lut.to_netcdf(STAT_LUT_FILENAME)
    duration = time.time() - start_time
    print(f"--- 阶段一完成。查找表已保存到 '{STAT_LUT_FILENAME}' (耗时: {duration:.2f} 秒) ---")
    return lut


def simulate_observation(true_yaw, true_tilt, empirical_model):
    """
    第二阶段：使用经验模型创建“模拟观测数据”。
    """
    print(f"\n--- 阶段二：为真实姿态 (Yaw={true_yaw}, Tilt={true_tilt}) 生成模拟观测数据 ---")
    true_l = np.random.uniform(MIN_L_CM, MAX_L_CM)
    print(f"  (模拟的随机体长: {true_l:.1f} cm)")
    
    ideal_angles = get_incidence_angles(true_yaw, true_tilt)
    absolute_ts = {}
    for beam_name in BEAM_NAMES:
        angle_with_uncertainty = ideal_angles[beam_name] + norm.rvs(scale=BEAM_UNCERTAINTY_STD)
        base_ts = empirical_model.get_ts(angle_with_uncertainty)
        length_effect = 20 * np.log10(true_l / REF_L_CM)
        final_ts = (base_ts + length_effect) + norm.rvs(scale=TS_NOISE_STD)
        absolute_ts[beam_name] = final_ts
        
    ts_v = absolute_ts['Vertical']
    ts_diff_obs = [absolute_ts['North']-ts_v, absolute_ts['South']-ts_v, absolute_ts['West']-ts_v, absolute_ts['East']-ts_v]
    
    print("TS差异观测值 (相对于Vertical):")
    for i, name in enumerate(BEAM_DIFF_NAMES):
        print(f"  - {name}: {ts_diff_obs[i]:.2f} dB")
    return ts_diff_obs
    

def run_bayesian_inversion(ts_diff_obs, stat_lut):
    """
    第三阶段：实现“贝叶斯解算器”。
    """
    print("\n--- 阶段三：开始运行贝叶斯解算器 ---")
    start_time = time.time()
    
    mu = stat_lut['ts_diff_mean'].values
    sigma = stat_lut['ts_diff_std'].values
    sigma = np.maximum(sigma, 1e-9)
    ts_obs_expanded = np.expand_dims(np.expand_dims(ts_diff_obs, axis=0), axis=0)
    
    log_likelihood_per_beam = -np.log(sigma) - (ts_obs_expanded - mu)**2 / (2 * sigma**2)
    total_log_likelihood = np.sum(log_likelihood_per_beam, axis=-1)

    log_prior = np.zeros_like(total_log_likelihood)
    log_posterior = total_log_likelihood + log_prior
    
    log_posterior -= np.max(log_posterior)
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior)
    
    duration = time.time() - start_time
    print(f"--- 阶段三完成。(耗时: {duration:.2f} 秒) ---")
    return posterior


def visualize_posterior(posterior_map, true_yaw, true_tilt, yaw_coords, tilt_coords):
    """
    第四阶段：结果可视化。
    """
    print("\n--- 阶段四：可视化后验概率分布 ---")
    max_prob_idx = np.unravel_index(np.argmax(posterior_map), posterior_map.shape)
    estimated_yaw, estimated_tilt = yaw_coords[max_prob_idx[0]], tilt_coords[max_prob_idx[1]]
    print(f"真实姿态: Yaw={true_yaw}, Tilt={true_tilt}")
    print(f"估计姿态: Yaw={estimated_yaw}, Tilt={estimated_tilt} (后验概率最高点)")

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(posterior_map.T, origin='lower', aspect='auto', extent=[yaw_coords[0], yaw_coords[-1], tilt_coords[0], tilt_coords[-1]], cmap='viridis')
    ax.set_title('Posterior Probability Distribution (using Empirical TS Model)')
    ax.set_xlabel('Yaw (degrees)'), ax.set_ylabel('Tilt (degrees)')
    fig.colorbar(im, ax=ax, label='Probability')
    ax.plot(true_yaw, true_tilt, 'r+', markersize=15, markeredgewidth=2, label='True State')
    ax.plot(estimated_yaw, estimated_tilt, 'wx', markersize=15, markeredgewidth=2, label='Estimated State (Max Prob)')
    ax.legend(), ax.grid(True, alpha=0.3)
    
    # 修改：将结果保存到 result/ 文件夹
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = "bayesian_inversion_empirical_result.png"
    save_path = os.path.join(RESULT_DIR, output_filename)
    plt.savefig(save_path)
    print(f"结果图已保存到 '{save_path}'")
    plt.show()


if __name__ == '__main__':
    # 新增：使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description='Bayesian solver for squid orientation using an empirical TS model.')
    parser.add_argument('--yaw', type=float, default=45, help='True yaw angle to simulate for the observation (in degrees).')
    parser.add_argument('--tilt', type=float, default=-20, help='True tilt angle to simulate for the observation (in degrees).')
    args = parser.parse_args()

    # 在开始前，删除旧的LUT文件以强制重新生成
    if os.path.exists(STAT_LUT_FILENAME):
        print(f"发现旧的经验模型查找表 '{STAT_LUT_FILENAME}'，将予以删除以强制重新生成。")
        os.remove(STAT_LUT_FILENAME)
        
    # 初始化基于真实数据的TS模型
    empirical_ts_model = EmpiricalTSModel(REAL_DATA_FILENAME, TARGET_FREQ_KHZ)

    # --- 阶段一：生成统计查找表 ---
    stat_lut = generate_statistical_lut(empirical_ts_model)

    # --- 阶段二：创建模拟观测 ---
    # 使用命令行传入的参数
    TRUE_YAW = args.yaw
    TRUE_TILT = args.tilt
    ts_observed = simulate_observation(TRUE_YAW, TRUE_TILT, empirical_ts_model)

    # --- 阶段三：运行贝叶斯解算器 ---
    posterior_distribution = run_bayesian_inversion(ts_observed, stat_lut)

    # --- 阶段四：可视化结果 ---
    visualize_posterior(posterior_distribution, TRUE_YAW, TRUE_TILT, YAW_ANGLES, TILT_ANGLES)
