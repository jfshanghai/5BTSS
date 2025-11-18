import matplotlib.pyplot as plt
import numpy as np

# --- 参数设置 ---
# 您可以修改这两个角度来看不同姿态下的鱿鱼位置
# 方向角 (Yaw, 0°=East, 90°=North)
psi_deg = 45
# 倾斜角 (Tilt, 正数=头朝上)
tau_deg = 20
# ---

# 将角度转换为弧度
psi = np.deg2rad(psi_deg)
tau = np.deg2rad(tau_deg)

# 创建3D绘图区域
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. 绘制世界坐标系轴
arrow_length = 1.5
ax.quiver(0, 0, 0, arrow_length, 0, 0, color='gray', arrow_length_ratio=0.1, label='East (+X)')
ax.quiver(0, 0, 0, 0, arrow_length, 0, color='gray', arrow_length_ratio=0.1, label='North (+Y)')
ax.quiver(0, 0, 0, 0, 0, arrow_length, color='gray', arrow_length_ratio=0.1, label='Up (+Z)')

# 2. 定义并绘制5个波束方向
beam_vectors = {
    'North #1': np.array([0, 1, 0]),
    'South #2': np.array([0, -1, 0]),
    'West #3': np.array([-1, 0, 0]),
    'East #4': np.array([1, 0, 0]),
    'Vertical #5': np.array([0, 0, -1])
}

beam_colors = {
    'North #1': 'cyan',
    'South #2': 'blue',
    'West #3': 'green',
    'East #4': 'lime',
    'Vertical #5': 'magenta'
}

for name, vec in beam_vectors.items():
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=beam_colors[name], arrow_length_ratio=0.1, label=name)

# 3. 计算鱿鱼的姿态向量
squid_vector = np.array([
    np.cos(tau) * np.cos(psi),
    np.cos(tau) * np.sin(psi),
    np.sin(tau)
])

# 4. 绘制鱿鱼姿态向量
# 将鱿鱼箭头向下移动，使其起始点在 Z=-0.5
squid_start_z = -0.5
ax.quiver(0, 0, squid_start_z, squid_vector[0], squid_vector[1], squid_vector[2],
          color='red', linewidth=2.5, arrow_length_ratio=0.15, label=f'Squid (ψ={psi_deg}°, τ={tau_deg}°)')

# 5. 计算并打印入射角
print("-" * 40)
print(f"Calculated incidence angles for squid at:")
print(f"  - Yaw (ψ): {psi_deg}°")
print(f"  - Tilt (τ): {tau_deg}°")
print("-" * 40)

for name, vec in beam_vectors.items():
    # 计算点积. 使用 np.clip 避免浮点误差导致 arccos 定义域问题
    dot_product = np.clip(np.dot(vec, squid_vector), -1.0, 1.0)
    
    # 计算角度（弧度），然后转换为度
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    
    print(f"Incidence angle for {name:<12}: {angle_deg:.2f} degrees")

print("-" * 40)


# 6. 设置图表样式
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('East-West Axis (X)')
ax.set_ylabel('North-South Axis (Y)')
ax.set_zlabel('Up-Down Axis (Z)')
ax.set_title('Echosounder Coordinate System & Squid Orientation')
ax.legend()
ax.grid(True)

# 设置视角
ax.view_init(elev=25, azim=-50)

# 保存并显示图表
plt.savefig('coordinate_system_visualization.png', dpi=300)
print("图表已保存为 'coordinate_system_visualization.png'")
plt.show()