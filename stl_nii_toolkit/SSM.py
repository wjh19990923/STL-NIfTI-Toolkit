import numpy as np
import open3d as o3d
import glob
import os
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transforms import find_transform
# ==============================
# 参数设置
# ==============================
folder = r"D:\kneefit_model_stl"
save_pkl = os.path.join(folder, "aligned_femurs_RE.pkl")  # 仅右腿
target_points = 10000  # 每个模型下采样点数
voxel_size = 1.5       # 控制下采样精度
n_components = 5        # PCA 模式数

# ==============================
# 1. 读取 femur 文件（只保留右腿 RE）
# ==============================
files = sorted(glob.glob(os.path.join(folder, "*Femur_RE*.stl")))
print(f"共找到 {len(files)} 个右腿 femur 模型")

if len(files) == 0:
    raise FileNotFoundError("❌ 没有找到包含 'Femur_RE' 的 STL 文件，请检查命名。")

# ==============================
# 2. 读取或重新生成对齐点云（使用 find_transform）
# ==============================
if os.path.exists(save_pkl):
    print(f"检测到已有对齐数据 {save_pkl}，直接加载。")
    aligned_points = pickle.load(open(save_pkl, "rb"))
else:
    aligned_points = []
    ref_file = files[0]  # 第一个 femur 作为参考
    ref_mesh = o3d.io.read_triangle_mesh(ref_file)
    ref_points = np.asarray(ref_mesh.sample_points_uniformly(number_of_points=target_points).points)
    aligned_points.append(ref_points)

    # 将参考点的坐标作为“固定模板点”
    ref_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_points))

    for i, src_file in enumerate(files[1:], start=2):
        print(f"\n🦴 对齐第 {i}/{len(files)} 个模型: {os.path.basename(src_file)}")
        T = find_transform(src_file, ref_file, number_of_points=20000)  # 高精度计算刚体变换

        src_mesh = o3d.io.read_triangle_mesh(src_file)
        src_mesh.transform(T)

        # 🚨 核心改动：使用“模板点位置”投影而非重新随机采样
        src_pcd = src_mesh.sample_points_uniformly(number_of_points=target_points)
        pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)

        matched_points = []
        for p in ref_points:  # 对于每个模板点，找 src_pcd 最近点
            _, idx, _ = pcd_tree.search_knn_vector_3d(p, 1)
            matched_points.append(src_pcd.points[idx[0]])

        aligned_points.append(np.asarray(matched_points))
        print(f"✅ 完成对齐并匹配到模板点 ({len(matched_points)} 点)")

    # 保存结果
    with open(save_pkl, "wb") as f:
        pickle.dump(aligned_points, f)
    print(f"✅ 已保存对齐数据到 {save_pkl}")



# ==============================
# 3. 确认点数一致并进行 PCA
# ==============================
num_points = aligned_points[0].shape[0]
assert all(pc.shape[0] == num_points for pc in aligned_points), "❌ 点数不一致！"

X = np.array([pc.flatten() for pc in aligned_points])
mean_shape = X.mean(axis=0)
pca = PCA(n_components=n_components)
pca.fit(X - mean_shape)

print("\n✅ PCA 完成")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"Mode {i+1}: {var*100:.2f}% 方差")

# ==============================
# 4. 保存结果
# ==============================
mean_shape_points = mean_shape.reshape(-1, 3)
mean_pcd = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(mean_shape_points))
mean_path = os.path.join(folder, "SSM_mean_femur_RE.ply")
o3d.io.write_point_cloud(mean_path, mean_pcd)
print(f"\n💾 平均形状已保存：{mean_path}")

# 保存 PCA 模型（方便之后加载）
np.savez(os.path.join(folder, "SSM_model_femur_RE.npz"),
         mean_shape=mean_shape,
         components=pca.components_,
         variance=pca.explained_variance_)
print("💾 PCA 模型已保存：SSM_model_femur_RE.npz")

# ==============================
# 5. 可视化第1主成分 ±3σ 变化
# ==============================
mode = 0
sigma = np.sqrt(pca.explained_variance_[mode])
shape_plus = mean_shape + 1 * sigma * pca.components_[mode]
shape_minus = mean_shape - 1 * sigma * pca.components_[mode]
plus_points = shape_plus.reshape(-1, 3)
minus_points = shape_minus.reshape(-1, 3)

mean_pcd.paint_uniform_color([0.8, 0.8, 0.8])   # 灰
plus_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plus_points))
plus_pcd.paint_uniform_color([1.0, 0.3, 0.3])   # 红
minus_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(minus_points))
minus_pcd.paint_uniform_color([0.3, 0.3, 1.0])  # 蓝

print("\n👀 显示平均形状、+3σ、-3σ（模式1变化）")
o3d.visualization.draw_geometries([mean_pcd], window_name="Mean Shape")
o3d.visualization.draw_geometries(
    [plus_pcd], window_name="Mean Shape + 1σ (Mode 1)")
o3d.visualization.draw_geometries(
    [minus_pcd], window_name="Mean Shape - 1σ (Mode 1)")

# ==============================
# 6. 动态演示函数：生成任意形状
# ==============================


def generate_shape(coeffs, mean_shape, pca):
    """根据主成分系数生成形状"""
    shape = mean_shape.copy()
    for i, c in enumerate(coeffs):
        shape += c * np.sqrt(pca.explained_variance_[i]) * pca.components_[i]
    return shape.reshape(-1, 3)


# 示例：生成随机形状
coeff_example = [1.2, -0.8, 0.5]
new_points = generate_shape(coeff_example, mean_shape, pca)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_points))
pcd.paint_uniform_color([0.3, 0.8, 0.4])
print("\n🎨 生成随机形状：coeff =", coeff_example)
o3d.visualization.draw_geometries([pcd], window_name="Generated Shape Random")

# ==============================
# 7. （可选）主成分系数分布
# ==============================
shape_params = pca.transform(X - mean_shape)
plt.figure()
plt.scatter(shape_params[:, 0], shape_params[:, 1], c='blue')
plt.xlabel("Mode 1")
plt.ylabel("Mode 2")
plt.title("Shape Parameter Distribution (Right Femur)")
plt.show()
