import open3d as o3d
import numpy as np

# 读取两个 STL 模型
source = o3d.io.read_triangle_mesh(r"C:\Users\wjh\Desktop\ETH\wjh-eth\git_sync\github_sync\STL-NIfTI-Toolkit\data\DPHK_01_femur_CORRECT.stl")
target = o3d.io.read_triangle_mesh(r"C:\Users\wjh\Desktop\ETH\wjh-eth\git_sync\github_sync\STL-NIfTI-Toolkit\data\DPHK_01_Femur.stl")


# 采样为点云
source_pcd = source.sample_points_uniformly(100000)
target_pcd = target.sample_points_uniformly(100000)

# 估计法线（必要）
source_pcd.estimate_normals()
target_pcd.estimate_normals()

# ✅ Step 1: 质心对齐
source_center = source_pcd.get_center()
target_center = target_pcd.get_center()
translation_init = target_center - source_center

# 构造初始刚体变换矩阵（平移）
T_init = np.eye(4)
T_init[:3, 3] = translation_init
source_pcd.transform(T_init)

# ✅ Step 2: ICP 精细配准
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd,
    max_correspondence_distance=10.0,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# 输出总变换（质心对齐 + ICP）
T_total = reg_p2p.transformation @ T_init

print("Final Transformation Matrix:")
print(T_total)

# 可视化对齐效果
source_pcd.transform(reg_p2p.transformation)  # 或直接用 T_total 再画一遍
o3d.visualization.draw_geometries([
    source_pcd.paint_uniform_color([1, 0, 0]),
    target_pcd.paint_uniform_color([0, 1, 0])
])

from scipy.spatial.transform import Rotation as R

# 提取旋转和平移
R_mat = T_total[:3, :3]
t_vec = T_total[:3, 3]

# 转为 Euler angles（根据需要选择旋转顺序，这里用 ZYX）
r = R.from_matrix(R_mat)
euler_deg = r.as_euler('zyx', degrees=True)

# 打印结果
print("\nTranslation vector (mm):")
print(t_vec)

print("\nEuler angles (degrees) [Z, Y, X]:")
print(euler_deg)
