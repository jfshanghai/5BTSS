# NetCDF (NC) 文件数据结构说明

本文档用于说明项目中使用的主要 NetCDF (`.nc`) 文件的数据结构，并提供如何使用 Python (`xarray`库) 读取和操作这些文件的示例。

## 1. `incidence_angle_lut_fixed.nc`

该文件是一个查找表 (Look-Up Table)，用于根据鱿鱼的姿态（方向角和倾斜角）快速查找五个声学波束的入射角。

### 数据结构

- **Dimensions**:
  - `yaw`: 360 (方向角, 0° to 359°)
  - `tilt`: 181 (倾斜角, -90° to 90°)
  - `beam`: 5 (声学波束名称)
- **Coordinates**:
  - `yaw (yaw)`: `[0, 1, ..., 359]` (单位: 度)
  - `tilt (tilt)`: `[-90, -89, ..., 90]` (单位: 度)
  - `beam (beam)`: `['North', 'South', 'West', 'East', 'Vertical']`
- **Data Variables**:
  - `__xarray_dataarray_variable__ (yaw, tilt, beam)`: 存储每个姿态组合对应的入射角 (单位: 度)。该变量是 `xarray` 打开单个 `DataArray` 并另存为 `Dataset` 时的默认名称。

### 使用示例

```python
import xarray as xr

# 打开查找表文件
try:
    lut_ds = xr.open_dataset('incidence_angle_lut_fixed.nc')
    # 提取主数据变量 (DataArray)
    lut = lut_ds['__xarray_dataarray_variable__']
    print("文件 'incidence_angle_lut_fixed.nc' 加载成功。")
except FileNotFoundError:
    print("错误: 'incidence_angle_lut_fixed.nc' 文件未找到。")
    # 退出或进行其他处理
    exit()


# --- 查询示例 ---
# 查询当方向角(yaw)为120°，倾斜角(tilt)为-30°时，所有波束的入射角
try:
    angles = lut.sel(yaw=120, tilt=-30, method='nearest')
    print("\n当 Yaw=120°, Tilt=-30° 时，各波束的入射角:")
    print(angles.to_pandas())
except KeyError:
    print("\n查询失败，请检查坐标名称和值是否正确。")

# 查询 'Vertical' 波束在 Yaw=45°, Tilt=20° 时的入射角
try:
    vertical_angle = lut.sel(yaw=45, tilt=20, beam='Vertical', method='nearest').item()
    print(f"\n当 Yaw=45°, Tilt=20° 时，'Vertical' 波束的入射角为: {vertical_angle:.2f}°")
except KeyError:
    print("\n查询失败，请检查坐标名称和值是否正确。")

# 关闭文件 (使用 with xr.open_dataset(...) 会自动关闭)
lut_ds.close()
```

---

## 2. `TS_interpolated_0p1kHz.nc`

该文件存储了两条鱿鱼在多个频率和角度下的目标强度 (Target Strength, TS) 插值数据。

### 数据结构

- **Dimensions**:
  - `squid_id`: 2 (鱿鱼编号)
  - `frequency`: 1201 (频率, 50-170 kHz, 步长 0.1 kHz)
  - `angle`: 83 (角度, -60° to 20°)
- **Coordinates**:
  - `squid_id (squid_id)`: `['No1', 'No2']`
  - `frequency (frequency)`: `[50.0, 50.1, ..., 170.0]` (单位: kHz)
  - `angle (angle)`: `[-60.0, -59.0, ..., 20.0]` (单位: 度)
- **Data Variables**:
  - `TS (squid_id, frequency, angle)`: 存储目标强度值 (单位: dB)

### 使用示例

```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# 打开插值后的TS数据文件
try:
    ts_ds = xr.open_dataset('TS_interpolated_0p1kHz.nc')
    print("\n文件 'TS_interpolated_0p1kHz.nc' 加载成功。")
except FileNotFoundError:
    print("错误: 'TS_interpolated_0p1kHz.nc' 文件未找到。")
    exit()

# --- 查询示例 ---
# 查询 'No1' 鱿鱼在频率为 70 kHz 和 120 kHz，角度为 0° 时的TS值
try:
    ts_values = ts_ds['TS'].sel(squid_id='No1', frequency=[70, 120], angle=0, method='nearest')
    print("\n'No1' 鱿鱼在 0° 角度，70kHz 和 120kHz 下的TS值:")
    print(ts_values.to_pandas())
except KeyError:
    print("\n查询失败，请检查坐标名称和值是否正确。")


# --- 绘图示例 ---
# 绘制 'No2' 鱿鱼在 120 kHz 时，TS随角度变化的曲线
try:
    ts_120k_no2 = ts_ds['TS'].sel(squid_id='No2', frequency=120, method='nearest')
    
    plt.figure(figsize=(10, 6))
    ts_120k_no2.plot(label='TS at 120 kHz for Squid No2')
    plt.title('TS vs. Angle for Squid No2 at 120 kHz')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Target Strength (dB)')
    plt.grid(True)
    plt.legend()
    # plt.show() # 取消注释以显示图像
    print("\n已生成 'No2' 鱿鱼在 120 kHz 的TS随角度变化图。")
except KeyError:
    print("\n绘图失败，请检查坐标名称和值是否正确。")

ts_ds.close()
```

---

## 3. `processed_squid_ts.nc`

该文件包含了原始处理后的鱿鱼TS数据，包含了两条鱿鱼在两个宽带系统（70kHz和120kHz中心频率）下的测量结果。

### 数据结构

- **Dimensions**:
  - `squid_id`: 2 (鱿鱼编号)
  - `system_freq`: 2 (声学系统中心频率)
  - `frequency`: 231 (每个系统内的频率点)
  - `angle`: 83 (角度)
- **Coordinates**:
  - `squid_id (squid_id)`: `['No1', 'No2']`
  - `system_freq (system_freq)`: `[70, 120]` (单位: kHz)
  - `frequency (frequency)`: `[50.0, 50.5, ..., 170.0]` (单位: kHz)
  - `angle (angle)`: `[-60.0, -59.0, ..., 20.0]` (单位: 度)
  - `ML_cm (squid_id)`: 鱿鱼的套膜长 (Mantle Length, 单位: cm)
  - `Weight_g (squid_id)`: 鱿鱼的重量 (单位: 克)
- **Data Variables**:
  - `TS (squid_id, system_freq, frequency, angle)`: 目标强度 (dB)
  - `TS_0_deg_std (squid_id, system_freq, frequency)`: 0度角时TS的标准差
- **Attributes**:
  - `description`: 数据的简要描述。
  - `source_files`: 原始数据源文件名列表。

### 使用示例

```python
import xarray as xr

# 打开处理后的TS数据文件
try:
    processed_ds = xr.open_dataset('processed_squid_ts.nc')
    print("\n文件 'processed_squid_ts.nc' 加载成功。")
except FileNotFoundError:
    print("错误: 'processed_squid_ts.nc' 文件未找到。")
    exit()

# --- 信息查询 ---
print("\n文件描述:")
# Note: The description may contain garbled characters if the encoding is not standard.
# It is recommended to check and fix the attribute in the source generation script if possible.
try:
    print(processed_ds.attrs['description'])
except Exception as e:
    print(f"Could not read description attribute: {e}")


print("\n鱿鱼信息:")
print(f"  套膜长 (cm): {processed_ds['ML_cm'].values}")
print(f"  重量 (g): {processed_ds['Weight_g'].values}")


# --- 数据提取示例 ---
# 提取 'No1' 鱿鱼在 70kHz 系统下，频率接近 68kHz，角度为-5°的数据
try:
    ts_data = processed_ds['TS'].sel(
        squid_id='No1', 
        system_freq=70, 
        frequency=68, 
        angle=-5, 
        method='nearest'
    )
    print(f"\n'No1' 鱿鱼在 70kHz 系统，约 68kHz，-5° 时的TS值为: {ts_data.item():.2f} dB")
except KeyError:
    print("\n数据提取失败，请检查坐标名称和值是否正确。")

processed_ds.close()
```