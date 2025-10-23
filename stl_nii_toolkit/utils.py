import pymeshlab
import trimesh
import os
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
from stl import mesh
import numpy as np
from scipy.spatial.transform import Rotation


def nii_check(corresponding_nii_path=None):
    # corresponding_nii_path = r"C:\Users\Public\Public Dupla\testbtoutput\anatomies\DPZM02\ana_000001.nii"
    # finally compare the corresponding nii file:
    original_shifted_nii = nib.load(corresponding_nii_path)
    print(original_shifted_nii)
    # get the data
    original_shifted_nii_data = original_shifted_nii.get_fdata()
    # print(original_shifted_nii_data)
    print(
        f'the maximum HU is {original_shifted_nii_data.max()}, AND the min HU is {original_shifted_nii_data.min()}')
    # assert original_shifted_nii_data.max() > 1
    # assert original_shifted_nii_data.min() == -1024

    from nibabel.orientations import aff2axcodes

    aff = original_shifted_nii.affine
    origin = aff[:3, 3]
    voxel_size = original_shifted_nii.header.get_zooms()[:3]
    axcodes = aff2axcodes(aff)  # e.g., ('L','P','S')

    print("Affine:\n", aff)
    print("Origin (world coord of voxel (0,0,0)):", origin)
    print("Voxel size (mm):", voxel_size)
    print("Axis codes:", axcodes)
    print("Units xyzt:", original_shifted_nii.header["xyzt_units"])


def recenter_nii(nii_path, out_path):
    import nibabel as nib
    import numpy as np
    import os

    # 输入 NIfTI 文件
    # nii_path = "your_image.nii.gz"
    # out_path = "your_image_recentered.nii.gz"

    img = nib.load(nii_path)
    affine = img.affine.copy()
    data = img.get_fdata()

    # 获取图像大小
    nx, ny, nz = data.shape[:3]

    # 计算图像中心的 voxel 坐标
    center_voxel = np.array([nx/2, ny/2, nz/2, 1])

    # 当前中心点在世界坐标的位置
    center_world = affine @ center_voxel

    # 把世界坐标的中心平移到原点(0,0,0)
    # 等价于在 affine[:3,3] 上减去中心点的世界坐标
    affine_recentered = affine.copy()
    affine_recentered[:3, 3] -= center_world[:3]

    # 创建新 NIfTI
    img_recentered = nib.Nifti1Image(
        data, affine_recentered, header=img.header)

    # 保存新文件
    nib.save(img_recentered, out_path)
    print(f"Saved recentered NIfTI to {out_path}")
    print("Old affine:\n", affine)
    print("New affine:\n", affine_recentered)


def align_nii_to_stl(nii_path, stl_path, out_path):
    nii = nib.load(nii_path)
    affine = nii.affine.copy()
    data = nii.get_fdata()

    # 获取 STL 边界
    mesh = trimesh.load(stl_path, force='mesh')
    bbox_min, bbox_max = mesh.bounds
    print("STL bounds:", bbox_min, bbox_max)

    # # === 1️⃣ LPS → RAS 坐标翻转 ===
    # flip = np.diag([-1, -1, 1, 1])
    # new_affine = flip @ affine @ flip  # 同时翻转 affine 的方向与平移项
    # print("Flipped affine (RAS):\n", new_affine)

    # === 2️⃣ 平移，让 NIfTI 原点贴合 STL ===
    nii_origin = affine[:3, 3]
    stl_origin = bbox_min  # 或者 mesh.centroid 视需求而定
    translation = stl_origin - nii_origin
    affine[:3, 3] = stl_origin

    print("NIfTI origin (after flip):", nii_origin)
    print("STL origin target:", stl_origin)
    # print("Translation applied:", translation)
    print("New affine:\n", affine)

    # === 3️⃣ 保存新的 NIfTI 文件 ===
    nii_new = nib.Nifti1Image(data, affine, header=nii.header)
    nib.save(nii_new, out_path)
    print(f"Aligned NIfTI saved to: {out_path}")


def compress_stl(input_stl, output_stl, target_ratio=0.5):
    """
    Compress the STL file with minimal loss, reducing the number of faces while preserving the shape.

    Parameters:
    - input_stl: Path to the input STL file
    - output_stl: Path to the output STL file
    - target_ratio: Target face ratio (0.0 - 1.0), e.g., 0.5 means reduce faces by 50%

    Returns:
    - None (the compressed STL file is saved)
    """
    # 1. read the STL file
    mesh = trimesh.load_mesh(input_stl)

    # 2. get the number of faces and calculate target face count
    original_faces = len(mesh.faces)
    target_faces = int(original_faces * target_ratio)

    print(f"Original faces: {original_faces}, Target faces: {target_faces}")

    # 3. use pymeshlab to create a MeshSet
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))

    # 4. simplfy without losing too much detail
    ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                    targetfacenum=30000,
                    preservenormal=True,
                    preservetopology=True,
                    planarquadric=True)

    # 5. output STL file
    ms.save_current_mesh(output_stl, binary=True)

    print(f"Compressed STL saved to: {output_stl}")


def stl_check(original_stl_path, downsampled_stl_path=None):
    """
    输出 STL 的 origin（质心），如果提供第二个 STL，则输出两者 origin 差。
    """
    # 读取原始 STL
    mesh_ori = trimesh.load(original_stl_path, force='mesh')
    if not isinstance(mesh_ori, trimesh.Trimesh):
        mesh_ori = trimesh.util.concatenate(mesh_ori.dump())
    origin_ori = mesh_ori.centroid

    print(f"Original STL origin (centroid): {origin_ori}")

    # 如果存在第二个 STL，则比较 origin 差
    if downsampled_stl_path:
        mesh_down = trimesh.load(downsampled_stl_path, force='mesh')
        if not isinstance(mesh_down, trimesh.Trimesh):
            mesh_down = trimesh.util.concatenate(mesh_down.dump())
        origin_down = mesh_down.centroid

        diff = origin_down - origin_ori
        print(f"Downsampled STL origin (centroid): {origin_down}")
        print(f"Origin difference (down - orig): {diff}")
        print(f"Difference magnitude: {np.linalg.norm(diff):.6f}")

    return origin_ori


def stl_origin_analysis(stl_path):
    mesh = trimesh.load(stl_path, force='mesh')
    vertices = mesh.vertices

    bbox_min, bbox_max = mesh.bounds
    centroid = mesh.centroid
    bbox_center = (bbox_min + bbox_max) / 2

    # 检查坐标范围
    print(f"STL Bounds:\n  Min: {bbox_min}\n  Max: {bbox_max}")
    print(f"Centroid: {centroid}")
    print(f"BBox center: {bbox_center}")

    # 检查是否包含 (0,0,0)
    contains_origin = np.all((bbox_min <= 0) & (bbox_max >= 0))
    print(f"Contains (0,0,0): {contains_origin}")

    # 如果包含原点，计算 (0,0,0) 相对中心的距离
    if contains_origin:
        dist_to_center = np.linalg.norm(centroid - np.array([0, 0, 0]))
        print(
            f"Distance from centroid to model origin (0,0,0): {dist_to_center:.3f} mm")
    else:
        print("Model does not include origin (0,0,0) in its bounding box.")

    return {
        "centroid": centroid,
        "bbox_center": bbox_center,
        "contains_origin": contains_origin
    }


def nii_add_air():
    # when a nii is produced from the segmentation pipeline (if done correctly), it is shifted and has a mask (0 or 1)
    # and a user should multiply that with the CT model (original file) to get a segmented nii file
    # but, in this way, the 0, which represent the air, does not have a HU of air, which should be -1024
    # TODO: add check if it has air value of -1024 and real HU unit (detecting values higher than 1)
    # file paths
    anatomy_path = r"\\hest.nas.ethz.ch\green_users_all_homes\wangjinh\MITK\DPZM_04\DPZM_04_tibia_shifted.nii"
    target_anatomy = r"\\hest.nas.ethz.ch\green_users_all_homes\wangjinh\MITK\DPZM_04\DPZM_04_shifted.nii"

    # load mask and target
    mask_img = nib.load(anatomy_path)
    target_img = nib.load(target_anatomy)

    # get data from the images
    mask_data = mask_img.get_fdata()
    target_data = target_img.get_fdata()

    # check if the shapes match
    if mask_data.shape != target_data.shape:
        raise ValueError(
            f"Shape mismatch: mask shape {mask_data.shape}, target shape {target_data.shape}")

    # apply mask
    # if mask_data = 0, masked_data = -1024
    # if mask_data = 1. masked_data = target_data
    masked_data = np.where(mask_data == 0, -1024, target_data)

    # create new NIfTI image
    masked_img = nib.Nifti1Image(
        masked_data, affine=mask_img.affine, header=mask_img.header)

    # save the masked image as a new NIfTI file to save space
    output_path = r"\\hest.nas.ethz.ch\green_users_all_homes\wangjinh\MITK\DPZM_04\DPZM_04_tibia_shifted_masked.nii"
    nib.save(masked_img, output_path)

    print(f"Masked anatomy saved to {output_path}")


print("trimesh version:", trimesh.__version__)


def compare_STL(input_stl, output_stl, sample_points=1000):
    """
    compare the difference between two STL file, including:
    - bounding box
    - centroid
    - extent (size)
    - rotation / translation difference using pointcloud resampling
    """
    mesh_input = trimesh.load(input_stl)
    mesh_output = trimesh.load(output_stl)

    print("===> Bounding Box:")
    print("Input Bounds :\n", mesh_input.bounds)
    print("Output Bounds:\n", mesh_output.bounds)

    center_input = mesh_input.centroid
    center_output = mesh_output.centroid
    print("\n===> Centroid (Center of Mass):")
    print("Input  Center :", center_input)
    print("Output Center :", center_output)
    print("Translation Vector:", center_output - center_input)

    size_input = mesh_input.extents
    size_output = mesh_output.extents
    print("\n===> Extents (Size along XYZ):")
    print("Input Size :", size_input)
    print("Output Size:", size_output)
    print("Scale ratio (Output/Input):", size_output / size_input)

    print("\n===> Estimating rigid transformation using sampled points:")
    try:
        # Sample equal number of points from both meshes
        points_input = mesh_input.sample(sample_points)
        points_output = mesh_output.sample(sample_points)

        # Procrustes analysis
        result = trimesh.registration.procrustes(
            points_input, points_output, reflection=False)

        matrix = result[0]
        residual = result[1]

        R = matrix[:3, :3]
        t = matrix[:3, 3]

        print("Rotation Matrix:\n", R)
        print("Translation    :", t)
        print("Residual Error :", residual)
        rot = Rotation.from_matrix(R)
        # transform the rotation matrix to Euler angles
        # also can be 'zyx', 'ZXY', etc.
        euler_angles_deg = rot.as_euler('xyz', degrees=True)
        print("Euler Angles (xyz, in degrees):", euler_angles_deg)
    except Exception as e:
        print("Failed to estimate transformation.")
        print("Error:", e)

    print("\nComparison complete.")


if __name__ == "__main__":
    # stl_check()
    # compare_STL(rf"C:\Users\Jinhao\Desktop\stl_examples\cr_fem_4_r_narrow_mm.stl",
    #              rf"C:\Users\Jinhao\Desktop\stl_examples\DPZM_02_Femur_42_5026_066_02_04_1.stl")
    # nii_check(corresponding_nii_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur.nii")
    # stl_check(
    #     original_stl_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur.stl",
    #     downsampled_stl_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur.stl"
    # )
    # recenter_nii(
    #     nii_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Tibia.nii",
    #     out_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Tibia_recentered.nii"
    # )

    # stl_origin_analysis(stl_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur.stl",)
    # nii_check(corresponding_nii_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur_recentered.nii")
    # align_nii_to_stl(
    #     nii_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur_recentered.nii",
    #     stl_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur.stl",
    #     out_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur_aligned.nii"
    # )
    stl_origin_analysis(stl_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Tibia.stl",)

    # nii_check(corresponding_nii_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\Bill_Li_Femur_aligned.nii")
    # stl_origin_analysis(stl_path=rf"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files\SUBN_02_Femur_RE_Surface.stl",)
    # nii_check(corresponding_nii_path=rf"D:\kneefit_model_nii\SUBN_02_Femur_RE_Volume.nii")
