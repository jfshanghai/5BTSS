#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Squid TS 实测数据 → NetCDF
修复点：把 frequency / angle 坐标强制转为 float，防止字符串维度导致
       sel(method='nearest') 时出现 str - str 报错
"""

import pandas as pd
import numpy as np
import xarray as xr
import os

def process_squid_ts_data():
    """
    读取 TS_squid 文件夹中的 4 个 Excel 表格，处理数据，
    并将所有数据整合到一个带有完整说明的 NetCDF 文件中。
    """
    print("开始处理鱿鱼 TS 实测数据...")

    # 1. 定义元数据
    squid_info = {
        'No1': {'ML_cm': 19.4, 'Weight_g': 132.2},
        'No2': {'ML_cm': 17.9, 'Weight_g': 137.6}
    }

    data_dir = 'TS_squid'
    file_list = [f for f in os.listdir(data_dir)
                 if f.endswith('.xlsx') and not f.startswith('~$')]

    all_data_arrays = []

    # 2. 循环处理每个文件
    for filename in file_list:
        print(f"正在处理文件: {filename}...")
        parts = filename.replace('.xlsx', '').split('_')
        system_freq_khz = int(parts[1])
        squid_id_str = parts[2]

        filepath = os.path.join(data_dir, filename)
        df = pd.read_excel(filepath, header=0, index_col=0)

        # 3. 处理重复的 0° 入射角
        zero_deg_cols = df.loc[:, 0]
        if isinstance(zero_deg_cols, pd.DataFrame) and not zero_deg_cols.empty:
            ts_0_deg_mean = zero_deg_cols.mean(axis=1)
            ts_0_deg_std = zero_deg_cols.std(axis=1)
            df = df.drop(columns=0)
            df[0] = ts_0_deg_mean
            df = df.sort_index(axis=1)
        else:
            ts_0_deg_std = pd.Series(0.0, index=df.index)

        # 4. 将 DataFrame 转为 xarray.DataArray
        # ------------------------------------------------------------------
        # 关键修复：把索引和列名统一转成 float，防止字符串坐标
        df.index = df.index.astype(float)
        df.columns = df.columns.astype(float)
        # ------------------------------------------------------------------
        sorted_angles = np.sort(df.columns.values)

        da = xr.DataArray(
            df.reindex(columns=sorted_angles).values,
            dims=('frequency', 'angle'),
            coords={
                'frequency': df.index.values,
                'angle': sorted_angles
            },
            name='TS_value'
        )

        da = da.expand_dims({
            'squid_id': [squid_id_str],
            'system_freq': [system_freq_khz]
        })

        # 0° 标准差
        da_std_0_deg = xr.DataArray(
            ts_0_deg_std.values,
            dims=('frequency'),
            coords={'frequency': ts_0_deg_std.index.astype(float).values},
            name='TS_0_deg_std_value'
        ).expand_dims({
            'squid_id': [squid_id_str],
            'system_freq': [system_freq_khz]
        })

        all_data_arrays.append({'ts': da, 'ts_0_deg_std': da_std_0_deg})

    # 5. 合并所有数据
    print("正在合并所有数据...")
    final_dataset = xr.Dataset()
    for item in all_data_arrays:
        ds_temp = xr.Dataset({
            'TS': item['ts'],
            'TS_0_deg_std': item['ts_0_deg_std']
        })
        final_dataset = final_dataset.merge(ds_temp, combine_attrs="override")

    # 统一维度顺序
    final_dataset['TS'] = final_dataset['TS'].transpose(
        'squid_id', 'system_freq', 'frequency', 'angle')
    final_dataset['TS_0_deg_std'] = final_dataset['TS_0_deg_std'].transpose(
        'squid_id', 'system_freq', 'frequency')

    # 6. 添加元数据
    final_dataset.attrs['description'] = (
        "鱿鱼(Squid)目标强度(TS)实测数据。"
        "包含两条鱿鱼在两个宽频系统下的测量结果。")
    final_dataset.attrs['source_files'] = str(file_list)

    final_dataset['squid_id'].attrs['description'] = "鱿鱼编号"
    final_dataset['system_freq'].attrs['description'] = "测量系统中心频率 (kHz)"
    final_dataset['frequency'].attrs['description'] = "实际测量频率 (kHz)"
    final_dataset['angle'].attrs['description'] = "声波入射角度 (度)"
    final_dataset['angle'].attrs['definition'] = (
        "0 degrees: Dorsal aspect (from back); "
        "-90 degrees: Tail aspect; "
        "90 degrees: Head aspect.")

    final_dataset['TS'].attrs['units'] = "dB"
    final_dataset['TS'].attrs['description'] = (
        "目标强度(Target Strength)。"
        "0度入射角的值是3次重复测量的平均值。")
    final_dataset['TS_0_deg_std'].attrs['units'] = "dB"
    final_dataset['TS_0_deg_std'].attrs['description'] = (
        "0度入射角下3次重复测量的标准差，可作为测量误差的估计。")

    # 7. 添加生物学信息
    final_dataset = final_dataset.assign_coords(
        ML_cm=('squid_id', [squid_info['No1']['ML_cm'], squid_info['No2']['ML_cm']]),
        Weight_g=('squid_id', [squid_info['No1']['Weight_g'], squid_info['No2']['Weight_g']])
    )
    final_dataset['ML_cm'].attrs['description'] = "鱿鱼胴长 (Mantle Length)"
    final_dataset['Weight_g'].attrs['description'] = "鱿鱼体重 (Weight)"

    # 8. 保存 NetCDF
    output_filename = 'processed_squid_ts_fix.nc'
    print(f"正在保存为 NetCDF 文件: {output_filename}...")
    final_dataset.to_netcdf(output_filename)
    print(f"处理完成！数据已保存到 {output_filename}")
    print("\n--- 数据集结构预览 ---")
    print(final_dataset)

if __name__ == '__main__':
    process_squid_ts_data()