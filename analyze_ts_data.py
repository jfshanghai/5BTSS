import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot():
    """
    加载和分析由 process_ts_data.py 脚本生成的 processed_squid_ts_fix.nc 文件，
    并生成三个示例图表。
    """
    try:
        file_path = 'processed_squid_ts.nc'
        ds = xr.open_dataset(file_path)
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{file_path}'。")
        print("请先运行 'process_ts_data.py' 来生成该文件。")
        return

    # --- 1. 打印NetCDF数据结构 ---
    print("--- 数据集结构预览 ---")
    print(ds)
    print("\n" + "="*50 + "\n")

    # --- 2. 绘图示例 A: TS vs. 入射角 ---
    print("正在生成图表1: TS vs. Angle...")
    try:
        SQUID_ID_A = 'No1'
        SYSTEM_FREQ_A = 120
        TARGET_FREQ_A = 120

        # 先精确匹配字符串维度，再对数值维度用 nearest
        ts_data_A = (
            ds['TS']
            .sel(squid_id=SQUID_ID_A, system_freq=SYSTEM_FREQ_A)
            .sel(frequency=TARGET_FREQ_A, method='nearest')
            .sortby('angle')
        )

        plt.figure(figsize=(12, 6))
        ts_data_A.plot.line(x='angle', marker='o', linestyle='-')
        plt.title(f'TS vs. Angle for Squid {SQUID_ID_A} @ {ts_data_A.frequency.item():.1f} kHz')
        plt.xlabel('Incidence Angle (degrees)')
        plt.ylabel('TS (dB)')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"绘制图表1时出错: {e}")

    # --- 3. 绘图示例 B: TS vs. 频率 (频率响应) ---
    print("正在生成图表2: TS vs. Frequency...")
    try:
        SQUID_ID_B = 'No1'
        SYSTEM_FREQ_B = 120
        TARGET_ANGLE_B = 0

        ts_data_B = (
            ds['TS']
            .sel(squid_id=SQUID_ID_B, system_freq=SYSTEM_FREQ_B)
            .sel(angle=TARGET_ANGLE_B, method='nearest')
            .sortby('frequency')
        )

        plt.figure(figsize=(12, 6))
        ts_data_B.plot.line(x='frequency', marker='.')
        plt.title(f'TS vs. Frequency for Squid {SQUID_ID_B} @ {ts_data_B.angle.item():.1f} degrees')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('TS (dB)')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"绘制图表2时出错: {e}")

    # --- 4. 绘图示例 C: 2D热力图 (TS vs. 频率 vs. 角度) ---
    print("正在生成图表3: 2D Heatmap...")
    try:
        SQUID_ID_C = 'No2'
        SYSTEM_FREQ_C = 70

        ts_data_C = (
            ds['TS']
            .sel(squid_id=SQUID_ID_C, system_freq=SYSTEM_FREQ_C)
            .sortby('angle')
            .sortby('frequency')
        )

        plt.figure(figsize=(12, 8))
        ts_data_C.plot.imshow(x='angle', y='frequency', cmap='viridis')
        plt.title(f'TS Heatmap for Squid {SQUID_ID_C} @ {SYSTEM_FREQ_C} kHz System')
        plt.xlabel('Incidence Angle (degrees)')
        plt.ylabel('Frequency (kHz)')
        plt.show()
    except Exception as e:
        print(f"绘制图表3时出错: {e}")


if __name__ == '__main__':
    analyze_and_plot()