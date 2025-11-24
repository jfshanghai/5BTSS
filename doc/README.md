# 鱿鱼姿态与声学目标强度（TS）模拟分析工具

## 概述

本项目提供了一套完整的工具链，用于处理、分析、模拟和可视化鱿鱼在五波束声呐探测下的声学目标强度（TS）数据。项目涵盖了从原始数据处理、声学模型建立、入射角查找表生成，到交互式姿态模拟和多维度数据可视化的全过程。

**核心功能:**
1.  **数据处理与插值**: 将离散的TS测量数据处理并插值为连续的NetCDF格式查找表。
2.  **入射角预计算**: 生成一个包含所有可能鱿鱼姿态（方向和倾斜角）下，五个声呐波束入射角的查找表（LUT），用于快速查询。
3.  **交互式姿态模拟**: 提供图形用户界面（GUI），实时可视化鱿鱼姿态、声呐波束，并计算对应的入射角。
4.  **TS数据可视化**: 提供多种脚本和GUI工具，用于绘制TS随角度变化的曲线、热力图等。
5.  **打包与分发**: 使用PyInstaller将GUI程序打包成独立的Windows可执行文件。

## 先决条件

在运行脚本之前，请确保您的Python环境中安装了以下库：
`pip install numpy pandas xarray netcdf4 matplotlib scipy pyinstaller tqdm`

## 重要配置：模型分辨率

**本项目核心脚本（`bayesian_solver.py`等）目前配置为1度分辨率。**

- **定义**: `YAW_ANGLES = np.arange(0, 360, 1)`, `TILT_ANGLES = np.arange(-60, 61, 1)`。
- **影响**: 高分辨率可以提高预测精度，但会**极大增加**查找表（LUT）的生成时间。如果您修改了分辨率，必须**删除旧的`ts_diff_empirical_lut.nc`文件并重新运行`bayesian_solver.py`**来生成新的LUT。

## 重要概念：入射角定义

**本项目统一采用 `[-90°, +90°]` 的入射角定义。**

- **物理意义**: 定义为声波传播方向与**垂直于鱿鱼身体长轴的平面**所成的角度。
- **具体定义**:
    - `+90°`: 声波从**头部**方向入射 (Head-on)。
    - `0°`: 声波与身体长轴**垂直**入射 (Broadside)。
    - `-90°`: 声波从**尾部**方向入射 (Tail-on)。

**文件与版本说明**:
*   本说明中提及的所有**最新脚本** (如 `create_lut_V1.1.py`, `squid_angle_gui.py`) 和它们生成的**最新数据文件** (文件名中带有 `_fix` 或 `_fixed` 的 `.nc` 文件) **均遵循此 `[-90°, +90°]` 定义**。
*   项目中可能存在一些未使用 ` - 90.0` 校正的早期脚本或数据文件（如 `V1.0` 版本），它们可能遵循 `[0°, 180°]` 的物理角度定义（0°=尾部, 180°=头部）。为确保一致性，请优先使用最新版本的脚本和数据。

## 主要文件说明

### 数据处理与生成

*   `process_ts_data.py`: 初版的TS数据处理脚本，用于读取 `TS_squid` 目录下的 `xlsx` 文件，进行插值并保存为 `TS_interpolated.nc`。
*   `process_ts_data_fix.py`: 修正版的数据处理脚本，解决了 `process_ts_data.py` 中的问题，并将结果保存为 `processed_squid_ts_fix.nc`。
*   `create_lut_V1.0.py` / `create_lut_V1.1.py`: 预计算声学入射角的脚本。它会遍历所有可能的鱿鱼姿态（方位角/俯仰角），计算五个波束的入射角，并将其存储在一个查找表（LUT）文件 `incidence_angle_lut.nc` 中，以加速后续计算。`V1.1` 为最新版本。

### 数据分析与可视化 (脚本)

*   `build_and_compare_ts_model.py`: **(核心模型构建脚本)** 用于根据真实的TS测量数据(`TS_interpolated_0p1kHz.nc`)构建和分析经验TS模型。功能包括：1. 可视化单频（如120kHz）下如何从原始数据构建插值模型；2. 可视化全频谱（频率-角度）的TS热力图；3. 直接对比70kHz和120kHz下“水平-垂直”TS差异曲线，以验证模型的频率依赖性。
*   `analyze_lut.py`: 用于分析和可视化 `ts_diff_statistical_lut.nc` 文件内容的脚本。它能打印出文件的内部结构，并绘制TS差异均值热力图。
*   `analyze_ts_data.py`: 用于分析 `processed_squid_ts_fix.nc` 文件的内容和结构。
*   `plot_coords.py`: 绘制三维坐标系、波束向量和鱿鱼姿态，用于调试和验证坐标系统。
*   `plot_ts_curves.py`: 从插值后的TS数据文件（`.nc`）中读取数据，绘制TS随倾角（Tilt）变化的曲线图，并保存为 `ts_vs_tilt_curves.png`。
*   `plot_ts_from_lut.py`: 演示如何使用 `incidence_angle_lut.nc` 和 `processed_squid_ts_fix.nc` 两个查找表，根据给定的鱿鱼姿态（Yaw/Tilt）查询对应的5个TS值，并绘制结果图，保存为 `ts_from_lut_plot.png`。
*   `plot_ts_heatmap_no_gui.py`: 读取 `processed_squid_ts_fix.nc` 文件，为五个波束分别生成TS强度关于鱿鱼姿态（Yaw/Tilt）的热力图，并保存为PNG图片。

### 模型反演与分析
*   `bayesian_solver.py`: 实现了基于贝叶斯方法的姿态反演模型。此版本采用**TS差异模型**，以消除体长对反演结果的影响。它包含完整的四个阶段：生成统计查找表、模拟观测、执行反演和可视化结果。
*   `fast_inversion_tester.py`: 用于快速测试贝叶斯反演算法的脚本。它会直接加载已生成的查找表（LUT），模拟一次观测并迅速给出反演结果，无需等待LUT的重新生成。
*   `batch_inversion_analysis.py`: **(新增)** 批量反演分析工具。可针对一个给定的姿态，重复运行N次模拟，将结果保存至 `result/` 目录下的CSV文件（默认），并进行统计分析和可视化，用于评估模型的稳定性和误差分布。支持引入先验知识来约束反演结果。

### 图形用户界面 (GUI)

*   `squid_angle_gui.py`: (中文版) 交互式姿态模拟工具。用户可以通过滑块调整鱿鱼的姿态，实时在3D图中查看姿态变化，并获取五个波束的入射角。
*   `squid_angle_gui_enV1.0.py` / `squid_angle_gui_enV1.1.py`: (英文版) 与中文版功能相同的交互式姿态模拟工具，界面语言为英文。`V1.1` 为最新版本。
*   `plot_ts_gui.py`: 交互式TS曲线查看器。允许用户通过滑块选择不同频率和方位角，动态查看TS随倾角变化的曲线。
*   `plot_ts_heatmap_gui.py`: 交互式TS热力图查看器。允许用户选择不同频率，显示五个波束的TS强度热力图。

### 数据文件 (`.nc`)

*   `incidence_angle_lut.nc`, `incidence_angle_lut_fixed.nc`: 存储鱿鱼在不同姿态下，五个声呐波束对应入射角的查找表。
*   `ts_statistical_lut.nc`: **(旧版本)** 存储通过蒙特卡洛方法生成的，用于贝叶斯反演的TS绝对值统计查找表（均值和标准差）。
*   `ts_diff_statistical_lut.nc`: **(旧版本)** 基于玩具模型生成的TS差异统计查找表。
*   `ts_diff_empirical_lut.nc`: **(最新版本)** 基于真实测量数据生成的TS差异统计查找表。
*   `TS_interpolated.nc`, `TS_interpolated_fixed.nc`, `processed_squid_ts.nc`, `processed_squid_ts_fix.nc`: 存储处理和插值后的鱿鱼TS数据，作为姿态解算的依据。

### 文档与配置

*   `README.md`: 本文件，项目说明。
*   `贝叶斯模拟流程说明.md`: **深入解释**了 `bayesian_solver.py` 脚本的端到端模拟与反演流程，包括**模型构建的两个层次、数据处理流程、结果解读以及后续研究步骤。**
*   `note.md`: 项目初期的笔记和思路。
*   `研究方法.md`: 定义坐标系、角度、向量和数值模拟流程。
*   `五维波束游泳状态解算.md`: 描述从TS值反演鱿鱼状态的数学方法。
*   `Q&A.md`: 项目工作流程梳理。
*   `squid_angle_gui.spec`: PyInstaller打包配置文件。
*   `.gitignore`: Git忽略文件配置。

## 使用说明

**注意**: 所有从此部分脚本生成的`.png`图片都将自动保存到项目根目录下的 `result/` 文件夹中。

### 1. (重要) 生成高分辨率查找表
**在执行任何分析之前，您必须先生成高分辨率的LUT文件。**
该脚本利用 `bayesian_solver.py` 生成`ts_diff_empirical_lut.nc`。**由于模型已更新为1度分辨率，首次运行此命令会耗费非常长的时间。**
```bash
# 运行一次，生成新的LUT文件
python bayesian_solver.py
```

### 2. (可选) 预处理数据
此步骤的目标是生成一个包含TS、角度、频率、鱿鱼ID等信息的NetCDF文件。
*   **处理原始TS数据**:
    ```bash
    python process_ts_data_fix.py
    ```

### 3. 构建和分析TS模型
基于预处理好的真实数据，分析并验证TS模型，确认其频率依赖性。
```bash
python build_and_compare_ts_model.py
```

### 4. 运行单次贝叶斯模拟
使用 `bayesian_solver.py` 对指定的姿态进行一次完整的模拟和反演。
```bash
# 指定一个新的“真实”姿态进行模拟
python bayesian_solver.py --yaw 90 --tilt 0
```

### 5. 分析查找表 (LUT)
运行此脚本可以查看 `ts_diff_empirical_lut.nc` 的内部结构，并将其中的数据分布绘制成热力图。
```bash
python analyze_lut.py
```

### 6. 快速测试反演算法
此脚本加载已有的`ts_diff_empirical_lut.nc`文件，快速测试不同姿态下的反演效果。您可以通过命令行参数方便地指定要测试的姿态。
```bash
# 测试 Yaw=120°, Tilt=-14°
python fast_inversion_tester.py --yaw 120 --tilt -14

# （新增）使用先验知识进行快速测试
python fast_inversion_tester.py --yaw 45 --tilt 10 --yaw-prior 10 170 --tilt-prior 0 60
```

### 7. (新增) 批量模拟与先验分析
使用 `batch_inversion_analysis.py` 评估模型在特定姿态下的稳定性和误差。
```bash
# 对 Yaw=45, Tilt=-20 运行100次模拟，并将结果保存到 a.csv
python batch_inversion_analysis.py --yaw 45 --tilt -20 --num-runs 100 --csv-path a.csv

# 引入先验知识：限定Yaw在[10,170]度，Tilt在[0,60]度
python batch_inversion_analysis.py --yaw 45 --tilt -20 --num-runs 50 --yaw-prior 10 170 --tilt-prior 0 60
```

### 8. 运行GUI工具
根据需要选择相应的GUI工具进行交互式分析。
*   **中文姿态模拟**:
    ```bash
    python squid_angle_gui.py
    ```
*   **TS热力图分析**:
    ```bash
    python plot_ts_heatmap_gui.py
    ```

### 9. PyInstaller打包
将GUI程序打包成 `.exe` 文件，便于在没有Python环境的机器上运行。
```bash
pyinstaller squid_angle_gui.spec
```
打包成功后，可执行文件位于 `dist/squid_angle_gui/` 目录下。
