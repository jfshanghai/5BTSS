# 鱿鱼姿态与声学入射角模拟工具

## 概述

本项目提供了一套工具，用于模拟和可视化五波束声呐探测鱿鱼时的声学入射角。主要功能包括：
1.  **预计算入射角查找表 (LUT)**：生成一个包含所有可能鱿鱼姿态（方向和倾斜角）下，五个声呐波束入射角的NetCDF文件，以便后续快速查询。
2.  **交互式GUI工具**：一个图形用户界面，用于实时可视化鱿鱼姿态、声呐波束，并计算对应的入射角。
3.  **PyInstaller打包配置**：将GUI程序打包成独立的Windows可执行文件。

## 先决条件

在运行脚本之前，请确保您的Python环境中安装了以下库：

```bash
pip install numpy xarray netcdf4 matplotlib pyinstaller
```

## 使用说明

### 1. 生成入射角查找表 (`create_lut.py`)

这个脚本用于预计算所有可能的鱿鱼姿态（方向和倾斜角）下，五个声呐波束的入射角，并将结果保存为NetCDF文件。这个文件可以作为后续模拟的查找表，避免重复计算。

**运行方式：**

```bash
python create_lut.py
```

**输出：**

*   成功运行后，会在当前目录下生成一个名为 `incidence_angle_lut.nc` 的NetCDF文件。
*   命令行会打印计算进度和耗时，并展示一个查找示例。

**后续使用：**

您可以在其他Python脚本中加载并查询这个查找表：

```python
import xarray as xr

# 加载查找表
lut = xr.open_dataarray('incidence_angle_lut.nc')

# 示例：查询 Yaw=123°, Tilt=-15° 时的入射角
squid_yaw = 123
squid_tilt = -15

# 使用 .sel() 方法快速查找对应的5个入射角
# method='nearest' 会自动找到最接近的坐标
incidence_angles = lut.sel(yaw=squid_yaw, tilt=squid_tilt, method='nearest')

print(f"Yaw={squid_yaw}°, Tilt={squid_tilt}° 的入射角:")
print(incidence_angles.values)
# 输出: [North_angle, South_angle, West_angle, East_angle, Vertical_angle]
```

### 2. 交互式GUI工具 (`squid_angle_gui.py`)

这个脚本提供了一个图形用户界面，您可以实时调整鱿鱼的方向角和倾斜角，并在三维图表中直观地看到其姿态变化，同时计算并显示每个波束的入射角。

**运行方式：**

```bash
python squid_angle_gui.py
```

**功能：**

*   **滑块调整**：拖动“方向角 (Yaw)”和“倾斜角 (Tilt)”滑块，图表和计算结果会实时更新。
*   **Z轴位置**：可以输入鱿鱼在Z轴上的位置，点击“更新图表和计算”按钮刷新。
*   **结果显示**：左侧文本框显示当前姿态下，五个波束的入射角。
*   **三维可视化**：右侧显示世界坐标系、五个波束方向和鱿鱼的姿态。

### 3. PyInstaller打包配置 (`squid_angle_gui.spec`)

这个文件是PyInstaller的配置文件，用于将 `squid_angle_gui.py` 打包成一个独立的Windows可执行文件 (`.exe`)。

**打包步骤：**

1.  **确保 `icon.ico` 文件存在**：将您希望作为程序图标的 `icon.ico` 文件放置在项目根目录下。
2.  **运行打包命令**：在命令行中，导航到项目根目录，然后执行：
    ```bash
    pyinstaller squid_angle_gui.spec
    ```
3.  **查找可执行文件**：打包成功后，您会在 `dist/squid_angle_gui/` 目录下找到 `squid_angle_gui.exe`。

**注意事项：**

*   `console=True` 是为了调试目的而设置的，它会在运行 `exe` 时打开一个命令行窗口。在最终发布版本中，您可以将 `squid_angle_gui.spec` 中的 `console=True` 改回 `console=False`，然后重新打包，以获得一个没有命令行窗口的GUI程序。
*   如果打包过程中遇到 `NameError: name 'collect_data_files' is not defined` 错误，请确保 `squid_angle_gui.spec` 文件顶部有 `from PyInstaller.utils.hooks import collect_data_files` 这一行。
*   如果运行 `exe` 时遇到 `_tkinter.TclError: bitmap "..." not defined` 错误，请确保 `squid_angle_gui.py` 中有 `resource_path` 函数和 `root.iconbitmap(resource_path('icon.ico'))` 调用，并且 `squid_angle_gui.spec` 的 `datas` 部分包含了 `('icon.ico', '.')`。

## 忽略文件 (`.gitignore`)

项目根目录下的 `.gitignore` 文件配置为忽略 `build/` 和 `dist/` 目录，这些是PyInstaller打包过程中生成的临时文件和输出目录，不应提交到版本控制。

## 项目文档

*   `README.md`: 本文件，项目的主要说明、文件结构和使用指南。
*   `note.md`: 项目初期的原始笔记和思路记录。
*   `研究方法.md`: 详细定义了本项目的坐标系、角度、向量，并规划了数值模拟的完整流程。
*   `五维波束游泳状态解算.md`: 详细描述了三种用于从5个TS实测值反演出鱿鱼游泳状态的数学方法（卡方、最大似然、贝叶斯）。
*   `Q&A.md`: 梳理和澄清了从建立模型到解算结果的完整工作流程，以问答形式呈现。
