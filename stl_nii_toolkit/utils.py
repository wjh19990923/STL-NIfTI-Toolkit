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


def nii_check():
    corresponding_nii_path = r"C:\Users\Public\Public Dupla\testbtoutput\anatomies\DPZM02\ana_000001.nii"
    # finally compare the corresponding nii file:
    original_shifted_nii = nib.load(corresponding_nii_path)
    print(original_shifted_nii)
    # get the data
    original_shifted_nii_data = original_shifted_nii.get_fdata()
    print(original_shifted_nii_data)
    print(f'the maximum HU is {original_shifted_nii_data.max()}, AND the min HU is {original_shifted_nii_data.min()}')
    assert original_shifted_nii_data.max() > 1
    assert original_shifted_nii_data.min() == -1024


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


def stl_check():
    # when a stl file is downsampled, its often get recentered so that the center of the stl does not match the original
    # this makes a coordinate difference between original stl and nii files, compared to the downsampled one
    # need to calculate this difference and add it to pose correction function
    # or should we transform our recentered stl file so that its has the same centre as the nii file.

    # file path
    original_stl_path = r"D:\MITK\Femur surface sigma 20.stl"
    downsampled_stl_path = r"D:\MITK\Femur surface sigma 20 python compress.stl"
    corresponding_nii_path = r"C:\Users\Public\Public Dupla\testbtoutput\anatomies\DPZM02\ana_000001.nii"
    # load STL file
    original_mesh = mesh.Mesh.from_file(original_stl_path)
    downsampled_mesh = mesh.Mesh.from_file(downsampled_stl_path)

    # obtain vertice coordinates
    original_vertices = original_mesh.vectors.reshape(-1, 3)  # (N, 3) 
    downsampled_vertices = downsampled_mesh.vectors.reshape(-1, 3)

    # breakpoint()
    # calculate center
    original_centroid = np.mean(original_vertices, axis=0)
    downsampled_centroid = np.mean(downsampled_vertices, axis=0)

    # calculate translational error
    translation_error = downsampled_centroid - original_centroid

    print("Original centroid:", original_centroid)
    print("Downsampled centroid:", downsampled_centroid)
    print("Translation error:", translation_error)

    # nii_check()
    # ---------------------------------------------


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
        raise ValueError(f"Shape mismatch: mask shape {mask_data.shape}, target shape {target_data.shape}")

    # apply mask
    # if mask_data = 0, masked_data = -1024
    # if mask_data = 1. masked_data = target_data 
    masked_data = np.where(mask_data == 0, -1024, target_data)

    # create new NIfTI image
    masked_img = nib.Nifti1Image(masked_data, affine=mask_img.affine, header=mask_img.header)

    # save the masked image as a new NIfTI file to save space
    output_path = r"\\hest.nas.ethz.ch\green_users_all_homes\wangjinh\MITK\DPZM_04\DPZM_04_tibia_shifted_masked.nii"
    nib.save(masked_img, output_path)

    print(f"Masked anatomy saved to {output_path}")

print("trimesh version:", trimesh.__version__)

import trimesh

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
        result = trimesh.registration.procrustes(points_input, points_output, reflection=False)

        matrix = result[0]
        residual = result[1]

        R = matrix[:3, :3]
        t = matrix[:3, 3]

        print("Rotation Matrix:\n", R)
        print("Translation    :", t)
        print("Residual Error :", residual)
        rot = Rotation.from_matrix(R)
        # transform the rotation matrix to Euler angles
        euler_angles_deg = rot.as_euler('xyz', degrees=True)  # also can be 'zyx', 'ZXY', etc.
        print("Euler Angles (xyz, in degrees):", euler_angles_deg)
    except Exception as e:
        print("Failed to estimate transformation.")
        print("Error:", e)

    print("\nComparison complete.")





if __name__ == "__main__":
    # stl_check()
    compare_STL(rf"C:\Users\Jinhao\Desktop\stl_examples\cr_fem_4_r_narrow_mm.stl",
                 rf"C:\Users\Jinhao\Desktop\stl_examples\DPZM_02_Femur_42_5026_066_02_04_1.stl")
