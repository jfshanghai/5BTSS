import pandas as pd
import numpy as np
import xarray as xr
import os

def process_squid_ts_data():
    """
    读取TS_squid文件夹中的4个Excel表格，处理数据，
    并将所有数据整合到一个带有完整说明的NetCDF文件中。
    """
    print("开始处理鱿鱼TS实测数据...")
    
    # 1. 定义元数据
    squid_info = {
        'No1': {'ML_cm': 19.4, 'Weight_g': 132.2},
        'No2': {'ML_cm': 17.9, 'Weight_g': 137.6}
    }
    
    data_dir = 'TS_squid'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    
    all_data_arrays = []

    # 2. 循环处理每个文件
    for filename in file_list:
        print(f"正在处理文件: {filename}...")
        
        # 从文件名解析信息
        parts = filename.replace('.xlsx', '').split('_')
        system_freq_khz = int(parts[1])
        squid_id_str = parts[2]
        
        # 读取Excel文件，第一行为表头，第一列为索引
        filepath = os.path.join(data_dir, filename)
        df = pd.read_excel(filepath, header=0, index_col=0)
        
        # 3. 处理重复的0°入射角
        # 找到所有列名为 0 的列
        zero_deg_cols = df.loc[:, 0]
        
        # 确保有多于一列的0°数据
        if isinstance(zero_deg_cols, pd.DataFrame) and not zero_deg_cols.empty:
            # 计算均值和标准差
            ts_0_deg_mean = zero_deg_cols.mean(axis=1)
            ts_0_deg_std = zero_deg_cols.std(axis=1)
            
            # 删除原始的0°列
            df = df.drop(columns=0)
            
            # 将均值作为新的0°列添加回去
            df[0] = ts_0_deg_mean
            
            # 重新排序，使0°在正确的位置
            df = df.sort_index(axis=1)
        else: # 如果只有一列0°或者没有，则标准差为0
            ts_0_deg_std = pd.Series(0.0, index=df.index)

        # 4. 将处理后的DataFrame转换为xarray.DataArray
        # 确保角度是排序的
        sorted_angles = np.sort(df.columns.values.astype(float))
        da = xr.DataArray(
            df.reindex(columns=sorted_angles).values, # Reindex df to match sorted angles
            dims=('frequency', 'angle'),
            coords={
                'frequency': df.index.values,
                'angle': sorted_angles
            },
            name='TS_value' # 指定DataArray的名称
        )
        
        # 添加本次处理的元数据
        da = da.expand_dims({
            'squid_id': [squid_id_str],
            'system_freq': [system_freq_khz]
        })
        
        # 为0°标准差也创建一个DataArray
        da_std_0_deg = xr.DataArray(
            ts_0_deg_std.values,
            dims=('frequency'),
            coords={'frequency': ts_0_deg_std.index.values},
            name='TS_0_deg_std_value' # 指定DataArray的名称
        ).expand_dims({
            'squid_id': [squid_id_str],
            'system_freq': [system_freq_khz]
        })
        
        all_data_arrays.append({'ts': da, 'ts_0_deg_std': da_std_0_deg})

    # 5. 合并所有数据
    print("正在合并所有数据...")
    
    # 创建一个空的Dataset来逐步填充数据
    final_dataset = xr.Dataset()

    for item in all_data_arrays:
        da_ts = item['ts']
        da_std = item['ts_0_deg_std']
        
        # 将每个DataArray转换为一个小的Dataset，然后合并
        ds_temp = xr.Dataset({
            'TS': da_ts,
            'TS_0_deg_std': da_std
        })
        
        # 使用merge来合并，xarray会自动处理不同坐标的对齐
        final_dataset = final_dataset.merge(ds_temp, combine_attrs="override")

    # 确保维度顺序一致
    # xarray.merge可能会改变维度顺序，这里统一一下
    if 'angle' in final_dataset.dims: # 只有TS有angle维度
        final_dataset['TS'] = final_dataset['TS'].transpose('squid_id', 'system_freq', 'frequency', 'angle')
    final_dataset['TS_0_deg_std'] = final_dataset['TS_0_deg_std'].transpose('squid_id', 'system_freq', 'frequency')

    # 6. 添加详细的元数据和说明
    final_dataset.attrs['description'] = (
        "鱿鱼(Squid)目标强度(TS)实测数据。"
        "包含两条鱿鱼在两个宽频系统下的测量结果。"
    )
    final_dataset.attrs['source_files'] = str(file_list)
    
    # 添加坐标说明
    final_dataset['squid_id'].attrs['description'] = "鱿鱼编号"
    final_dataset['system_freq'].attrs['description'] = "测量系统中心频率 (kHz)"
    final_dataset['frequency'].attrs['description'] = "实际测量频率 (kHz)"
    final_dataset['angle'].attrs['description'] = "声波入射角度 (度)"
    final_dataset['angle'].attrs['definition'] = (
        "0 degrees: Dorsal aspect (from back); "
        "-90 degrees: Tail aspect; "
        "90 degrees: Head aspect."
    )
    
    # 添加数据变量说明
    final_dataset['TS'].attrs['units'] = "dB"
    final_dataset['TS'].attrs['description'] = (
        "目标强度(Target Strength)。"
        "0度入射角的值是3次重复测量的平均值。"
    )
    final_dataset['TS_0_deg_std'].attrs['units'] = "dB"
    final_dataset['TS_0_deg_std'].attrs['description'] = (
        "0度入射角下3次重复测量的标准差，可作为测量误差的估计。"
    )
    
    # 添加鱿鱼生物学信息
    final_dataset = final_dataset.assign_coords(
        ML_cm=('squid_id', [squid_info['No1']['ML_cm'], squid_info['No2']['ML_cm']]),
        Weight_g=('squid_id', [squid_info['No1']['Weight_g'], squid_info['No2']['Weight_g']])
    )
    final_dataset['ML_cm'].attrs['description'] = "鱿鱼胴长 (Mantle Length)"
    final_dataset['Weight_g'].attrs['description'] = "鱿鱼体重 (Weight)"

    # 7. 保存为NetCDF文件
    output_filename = 'processed_squid_ts.nc'
    print(f"正在保存为NetCDF文件: {output_filename}...")
    final_dataset.to_netcdf(output_filename)
    
    print(f"处理完成！数据已保存到 {output_filename}")
    print("\n--- 数据集结构预览 ---")
    print(final_dataset)

if __name__ == '__main__':
    process_squid_ts_data()
