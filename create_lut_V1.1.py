import numpy as np
import xarray as xr
import time

def create_incidence_angle_lut():
    """
    计算所有可能的鱿鱼姿态下的声学入射角，
    并生成一个查找表（LUT），保存为NetCDF文件。
    """
    print("开始计算入射角查找表 (LUT)...")
    start_time = time.time()

    # 1. 定义姿态角坐标
    # 方向角 (Yaw ψ): 0-359°, 步长1°
    yaw_angles = np.arange(0, 360, 1)
    # 倾斜角 (Tilt τ): -90-90°, 步长1°
    tilt_angles = np.arange(-90, 91, 1)
    beam_names = ['North', 'South', 'West', 'East', 'Vertical']

    # 2. 定义波束向量
    beam_vectors = {
        'North': np.array([0, 1, 0]),
        'South': np.array([0, -1, 0]),
        'West': np.array([-1, 0, 0]),
        'East': np.array([1, 0, 0]),
        'Vertical': np.array([0, 0, -1])
    }

    # 3. 使用向量化计算以提高效率
    # 创建一个包含所有(yaw, tilt)组合的网格
    psi_rad, tau_rad = np.meshgrid(np.deg2rad(yaw_angles), np.deg2rad(tilt_angles))

    # 一次性计算所有姿态组合的鱿鱼身体向量 S
    # S_x, S_y, S_z 的形状都将是 (181, 360)
    S_x = np.cos(tau_rad) * np.cos(psi_rad)
    S_y = np.cos(tau_rad) * np.sin(psi_rad)
    S_z = np.sin(tau_rad)

    # 将它们堆叠成一个多维向量数组，形状为 (181, 360, 3)
    squid_vectors = np.stack([S_x, S_y, S_z], axis=-1)

    # 4. 为每个波束计算入射角
    # 创建一个空数组用于存储结果，形状为 (181, 360, 5)
    incidence_angles_deg = np.zeros((len(tilt_angles), len(yaw_angles), len(beam_names)))

    for i, beam_name in enumerate(beam_names):
        beam_vec = beam_vectors[beam_name]
        
        # 使用爱因斯坦求和约定(einsum)高效计算整个网格的点积
        dot_product = np.einsum('ijk,k->ij', squid_vectors, beam_vec)
        
        # 为防止浮点误差导致定义域问题，裁剪数值
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 计算角度并存入结果数组
        incidence_angles_deg[:, :, i] = np.rad2deg(np.arccos(dot_product)) - 90.0

    # 5. 创建一个带标签的 xarray.DataArray
    # 我们将维度顺序调整为 (yaw, tilt, beam) 以方便直觉
    lut = xr.DataArray(
        data=np.transpose(incidence_angles_deg, (1, 0, 2)), # 转换维度
        coords={
            'yaw': yaw_angles,
            'tilt': tilt_angles,
            'beam': beam_names
        },
        dims=('yaw', 'tilt', 'beam'),
        attrs={
            'description': 'Lookup table of acoustic incidence angles for a 5-beam echosounder.',
            'units': 'degrees',
            'yaw_definition': '0-359 degrees, 0 = East, 90 = North',
            'tilt_definition': '-90 to +90 degrees, positive = head up'
        }
    )

    # 6. 保存为 NetCDF 文件
    output_filename = 'incidence_angle_lut_fixed.nc'
    print(f"计算完成。正在保存数据到 '{output_filename}'...")
    lut.to_netcdf(output_filename)

    end_time = time.time()
    print(f"成功创建 '{output_filename}'。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")
    
    # 7. 打印一个使用示例
    print("\n--- 使用示例 ---")
    # 查找 Yaw=45°, Tilt=20° 时的入射角
    example_lookup = lut.sel(yaw=45, tilt=20, method='nearest')
    print(f"当 Yaw=45°, Tilt=20° 时，入射角为:")
    print(example_lookup.to_pandas())

if __name__ == '__main__':
    create_incidence_angle_lut()
