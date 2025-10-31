import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def find_transform(source_stl: str, target_stl: str, number_of_points: int = 100000, visualize: bool = False) -> np.ndarray:
    """
    给定两个 STL 文件，计算将 source 配准到 target 的刚性变换矩阵。
    返回 4x4 的变换矩阵。
    """
    # 读取 STL 网格并转换为点云
    source = o3d.io.read_triangle_mesh(source_stl)
    target = o3d.io.read_triangle_mesh(target_stl)

    source_pcd = source.sample_points_uniformly(
        number_of_points=number_of_points)
    target_pcd = target.sample_points_uniformly(
        number_of_points=number_of_points)

    source_pcd.estimate_normals()
    target_pcd.estimate_normals()

    # Step 1: 质心对齐
    T_init = np.eye(4)
    T_init[:3, 3] = target_pcd.get_center() - source_pcd.get_center()
    source_pcd.transform(T_init)

    # Step 2: ICP 精配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=10.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=2000)
    )

    # 合并初始对齐与 ICP 变换
    T_total = reg_p2p.transformation @ T_init

    # 输出结果
    print("✅ Final Transformation Matrix:\n", T_total)

    # 分解旋转和平移
    R_mat = T_total[:3, :3]
    t_vec = T_total[:3, 3]
    euler_deg = R.from_matrix(R_mat).as_euler('zyx', degrees=True)

    print("\n📦 Translation vector (mm):", np.round(t_vec, 2))
    print("🎯 Euler angles (degrees) [Z, Y, X]:", np.round(euler_deg, 2))

    # 可视化配准效果
    if visualize:
        source_pcd.transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries([
            source_pcd.paint_uniform_color([1, 0, 0]),
            target_pcd.paint_uniform_color([0, 1, 0])
        ])

    return T_total


def apply_transform(mesh_path: str, T: np.ndarray, output_path: str):
    """
    将给定的变换矩阵 T 应用到 mesh，并保存为新的 STL。
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.transform(T)
    mesh.compute_vertex_normals()  # ✅ 关键：必须有法线才能写 STL
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"✅ Saved transformed mesh to: {output_path}")


if __name__ == "__main__":
    # 配准输入
    source_path = r"C:\Users\wjh\Desktop\ETH\wjh-eth\git_sync\github_sync\STL-NIfTI-Toolkit\data\DPHK_01_femur_CORRECT.stl"
    target_path = r"C:\Users\wjh\Desktop\ETH\wjh-eth\git_sync\github_sync\STL-NIfTI-Toolkit\data\DPHK_01_Femur.stl"

    # 应用变换后保存路径
    output_path = r"C:\Users\wjh\Desktop\ETH\wjh-eth\git_sync\github_sync\STL-NIfTI-Toolkit\data\DPHK_01_Femur_CORRECT_transformed.stl"

    # 配准并获得变换
    T_final = find_transform(source_path, target_path)

    # 应用变换并保存
    apply_transform(source_path, T_final, output_path)
