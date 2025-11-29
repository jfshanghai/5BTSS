import pandas as pd
import os

def process_and_export_csv():
    # 定义任务映射：源文件名 -> (目标频率, 输出文件名)
    # 注意：源文件在 TS_squid 目录下
    tasks = [
        {'src': 'TS_70_No1.xlsx',  'freq': 70,  'out': 'Squid01_TS_measured_70kHz.csv'},
        {'src': 'TS_120_No1.xlsx', 'freq': 120, 'out': 'Squid01_TS_measured_120kHz.csv'},
        {'src': 'TS_70_No2.xlsx',  'freq': 70,  'out': 'Squid02_TS_measured_70kHz.csv'},
        {'src': 'TS_120_No2.xlsx', 'freq': 120, 'out': 'Squid02_TS_measured_120kHz.csv'}
    ]

    base_dir = 'TS_squid'
    output_dir = 'result' # 新增：输出目录
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始提取中心频率TS数据并导出为CSV...")

    for task in tasks:
        src_path = os.path.join(base_dir, task['src'])
        target_freq = task['freq']
        out_file_name = task['out'] # 原始文件名
        out_file_path = os.path.join(output_dir, out_file_name) # 完整的输出路径

        if not os.path.exists(src_path):
            print(f"错误: 找不到文件 {src_path}")
            continue

        print(f"正在处理: {src_path} -> 提取 {target_freq}kHz -> {out_file_path}")

        try:
            # 读取Excel，第一列为频率索引，第一行为角度
            df = pd.read_excel(src_path, header=0, index_col=0)

            # --- 1. 处理0度角的重复测量 (计算平均值) ---
            # 检查列名中是否存在 0 (pandas读取excel列名可能是整数0或浮点数0.0)
            zero_cols = [c for c in df.columns if c == 0 or c == 0.0]
            
            if len(zero_cols) > 0:
                # 提取所有0度列
                df_zero = df[zero_cols]
                # 计算行平均值
                mean_zero = df_zero.mean(axis=1)
                # 删除原始的0度列
                df = df.drop(columns=zero_cols)
                # 插入平均值作为新的0度列
                df[0] = mean_zero
            
            # --- 2. 提取指定中心频率的数据 ---
            # 尝试查找索引中的频率
            if target_freq in df.index:
                ts_series = df.loc[target_freq]
            else:
                # 如果精确匹配失败（例如浮点数精度问题），尝试找最近的
                print(f"  警告: 在索引中未找到精确的 {target_freq}kHz，尝试查找最近的频率...")
                nearest_freq = df.index[abs(df.index - target_freq).argmin()]
                print(f"  找到最近频率: {nearest_freq}kHz")
                ts_series = df.loc[nearest_freq]

            # --- 3. 格式化数据 ---
            # ts_series 的 index 是角度 (columns of df), value 是 TS
            out_df = pd.DataFrame({
                'angle_deg': ts_series.index,
                'ts_db': ts_series.values
            })

            # 确保角度是数值型并排序
            out_df['angle_deg'] = pd.to_numeric(out_df['angle_deg'])
            out_df = out_df.sort_values(by='angle_deg')

            # --- 4. 保存CSV ---
            out_df.to_csv(out_file_path, index=False)
            print(f"  已保存: {out_file_path} (行数: {len(out_df)})")

        except Exception as e:
            print(f"  处理 {src_path} 时发生错误: {e}")

    print("\n所有任务完成。\n")

if __name__ == '__main__':
    process_and_export_csv()
