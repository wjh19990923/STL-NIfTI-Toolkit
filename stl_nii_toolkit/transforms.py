import open3d as o3d
import numpy as np

# 读取两个 STL 模型
source = o3d.io.read_triangle_mesh("source.stl")
target = o3d.io.read_triangle_mesh("target.stl")

# 转换为点云用于配准（可以是顶点或采样）
source_pcd = source.sample_points_uniformly(number_of_points=10000)
target_pcd = target.sample_points_uniformly(number_of_points=10000)

# 初始对齐（中心化）
source_pcd.estimate_normals()
target_pcd.estimate_normals()

# ICP 刚体配准
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, max_correspondence_distance=5.0,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# 输出变换矩阵
print("Transformation Matrix:")
print(reg_p2p.transformation)

# 可视化对齐结果
source_pcd.transform(reg_p2p.transformation)
o3d.visualization.draw_geometries([source_pcd.paint_uniform_color([1, 0, 0]),
                                   target_pcd.paint_uniform_color([0, 1, 0])])
