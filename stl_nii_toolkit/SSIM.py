import os
import glob
import numpy as np
import nibabel as nib
import scipy.ndimage
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ==============================
# 工具函数
# ==============================
def resample_to_shape(volume, target_shape=(128, 128, 128)):
    """重采样到统一形状"""
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return scipy.ndimage.zoom(volume, zoom_factors, order=1)


def normalize_intensity(volume):
    """强度归一化到 [0, 1]"""
    # vmin, vmax = np.percentile(volume, (1, 99))
    vmin, vmax = -1024, 1024
    # volume = np.clip(volume, vmin, vmax)
    return (volume - vmin) / (vmax - vmin + 1e-8)


def apply_affine_centered(volume, matrix):
    """以中心为旋转中心进行仿射变换（仅旋转）"""
    center = np.array(volume.shape) / 2.0
    offset = center - matrix @ center
    aligned = scipy.ndimage.affine_transform(
        volume, matrix=matrix, offset=offset,
        order=1, mode="constant", cval=0.0
    )
    return aligned


def rigid_align_translation(moving_vol, ref_vol, threshold=0.1):
    """仅基于质心的平移配准"""
    moving_mask = (moving_vol > threshold).astype(np.float32)
    ref_mask = (ref_vol > threshold).astype(np.float32)

    moving_center = np.array(scipy.ndimage.center_of_mass(moving_mask))
    ref_center = np.array(scipy.ndimage.center_of_mass(ref_mask))

    shift = ref_center - moving_center
    aligned = scipy.ndimage.shift(
        moving_vol, shift=shift, order=1, mode='constant', cval=0.0
    )
    return aligned, shift


def get_icp_rotation(moving_vol, ref_vol, threshold=0.1, max_iter=50):
    """仅提取 ICP 旋转矩阵（含 Open3D→NumPy 坐标系修正）"""
    moving_pts = np.argwhere(moving_vol > threshold)
    ref_pts = np.argwhere(ref_vol > threshold)

    if len(moving_pts) < 100 or len(ref_pts) < 100:
        print("⚠️ 点数太少，跳过 ICP")
        return np.eye(3)

    pcd_moving = o3d.geometry.PointCloud()
    pcd_ref = o3d.geometry.PointCloud()
    pcd_moving.points = o3d.utility.Vector3dVector(moving_pts)
    pcd_ref.points = o3d.utility.Vector3dVector(ref_pts)

    # ICP 配准
    reg = o3d.pipelines.registration.registration_icp(
        pcd_moving, pcd_ref,
        max_correspondence_distance=10.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter),
    )

    R_o3d = reg.transformation[:3, :3]

    # ✅ 坐标系修正：Open3D(右手, y↑, z前) → NumPy(体素, y↓, z内)
    C = np.diag([1, -1, -1])
    R = C @ R_o3d @ C

    return R


def visualize_mode(mean_vol, plus_vol, minus_vol, mode_idx):
    """可视化 ±3σ"""
    slice_idx = mean_vol.shape[2] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(mean_vol[:, :, slice_idx], cmap="gray")
    plt.title(f"Mean (Mode {mode_idx+1})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(plus_vol[:, :, slice_idx], cmap="gray")
    plt.title(f"+3σ (Mode {mode_idx+1})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(minus_vol[:, :, slice_idx], cmap="gray")
    plt.title(f"−3σ (Mode {mode_idx+1})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==============================
# 主函数
# ==============================
def main():
    folder = r"D:\kneefit_model_nii"
    cache_dir = os.path.join(folder, "resampled_cache_icp_final")
    os.makedirs(cache_dir, exist_ok=True)

    pattern = os.path.join(folder, "*Femur_RE*.nii*")
    files = sorted(glob.glob(pattern))
    print(f"✅ 找到 {len(files)} 个 Femur NIfTI 文件")
    if len(files) == 0:
        raise FileNotFoundError("❌ 没有找到任何 *Femur_RE*.nii 文件。")

    target_shape = (128, 128, 128)
    n_components = 5

    # ==============================
    # 1️⃣ Normalize + Resample (带缓存)
    # ==============================
    resampled_volumes = []
    for f in files:
        name = os.path.basename(f).replace(".nii", "").replace(".gz", "")
        cache_path = os.path.join(cache_dir, f"{name}_resampled.npy")

        if os.path.exists(cache_path):
            vol = np.load(cache_path)
            print(f"⚡ 已加载缓存: {os.path.basename(cache_path)}")
        else:
            nii = nib.load(f)
            vol = nii.get_fdata().astype(np.float32)
            vol = normalize_intensity(vol)
            vol = resample_to_shape(vol, target_shape)
            np.save(cache_path, vol)
            print(f"💾 重采样并缓存: {name}")

        resampled_volumes.append(vol)

    # ==============================
    # 2️⃣ 旋转 + 平移配准
    # ==============================
    ref_vol = resampled_volumes[0]
    aligned_volumes = []

    for i, moving_vol in enumerate(resampled_volumes):
        if i == 0:
            aligned = ref_vol
            print("🦴 使用第一个样本作为参考体积")
        else:
            print(f"\n🦴 第 {i+1}/{len(resampled_volumes)} 个体积配准中...")
            R = get_icp_rotation(moving_vol, ref_vol)
            rotated = apply_affine_centered(moving_vol, R)
            aligned, shift = rigid_align_translation(rotated, ref_vol)
            print(f"✅ ICP旋转 + 质心平移完成，平移向量={shift}")

        aligned_volumes.append(aligned)

    X = np.stack([v.flatten() for v in aligned_volumes], axis=0)
    print(f"\n✅ 旋转 + 平移配准完成，矩阵维度: {X.shape}")

    # ==============================
    # 3️⃣ PCA 建模
    # ==============================
    print("\n🚀 执行 PCA 建模 ...")
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("✅ PCA 完成\n")

    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"Mode {i+1}: {var*100:.2f}% 方差")

    mean_volume = pca.mean_.reshape(target_shape)

    # ==============================
    # 4️⃣ 可视化 ±3σ 模式变化
    # ==============================
    for i in range(n_components):
        sigma = np.sqrt(pca.explained_variance_[i])
        mode_plus = (pca.mean_ + 3 * sigma *
                     pca.components_[i]).reshape(target_shape)
        mode_minus = (pca.mean_ - 3 * sigma *
                      pca.components_[i]).reshape(target_shape)
        print(
            f"\n🎨 Mode {i+1}: 可视化 ±3σ (方差占比: {pca.explained_variance_ratio_[i]*100:.2f}%)")
        visualize_mode(mean_volume, mode_plus, mode_minus, i)

    print("\n✅ ICP旋转(坐标系修正) + 平移 + PCA 建模完成！")


if __name__ == "__main__":
    main()
