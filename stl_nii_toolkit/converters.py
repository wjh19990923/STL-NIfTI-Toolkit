import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import vtk
from vtk.util import numpy_support
import nibabel as nib
from trimesh import proximity
from scipy.spatial.transform import Rotation as R
from function_tools import *
# dupla scene tools
from dupla_renderers.pytorch3d import (
    AnatomyCT,
    AnatomySTL,
    Camera,
    CTRenderer,
    CubeCT,
    CubeSTL,
    Scene,
    STLRenderer,
    anatomy_ct_to_anatomy_stl,
)

DTYPE_TORCH = torch.float32
from numpy.linalg import inv
from scipy.signal import savgol_filter
import vtk
from vtk.util import numpy_support
import numpy as np
import nibabel as nib


# 使用提供的函数转换STL到NIfTI
def give_contact_num(ray):
    contact_count = 0
    for i in range(len(ray)):
        val = ray[i]
        if val > 0:
            contact_count += 1

    return contact_count
def filling_stl(output_nifti_path,HU=10000):
    img = nib.load(output_nifti_path)
    data = img.get_fdata()
    folder, filename = os.path.split(output_nifti_path)
    # 移除文件扩展名（.stl）
    base_filename = os.path.splitext(filename)[0]
    test_image_folder = rf'data_files/anatomies/test_images/{base_filename}'
    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder, exist_ok=True)
    for i in range(data.shape[0]):
        image = data[i, :, :]
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                voxel = image[j, k]
                if voxel < 0:
                    ray1 = data[i, j, :k]
                    ray2 = data[i, j, k:]
                    ray3 = data[i, :j, k]
                    ray4 = data[i, j:, k]

                    num_voxels_1 = give_contact_num(ray1)
                    num_voxels_2 = give_contact_num(ray2)
                    num_voxels_3 = give_contact_num(ray3)
                    num_voxels_4 = give_contact_num(ray4)
                    if num_voxels_1 >= 1 and num_voxels_2 >= 1 and num_voxels_3 >= 1 and num_voxels_4 >= 1:
                        image[j, k] = HU
                        data[i, j, k] = HU
        plt.imsave(os.path.join(test_image_folder,f'layer{i}.png'), image, cmap='gray')
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    new_output_filename = output_nifti_path.replace("nofill", "filled")
    nib.save(new_img, new_output_filename)
    return new_output_filename

def stl_to_nifti(stl_path, output_nifti_path=None, HU=10000, voxel_size=(1, 1, 1)):
    # 读取STL网格
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    polydata = reader.GetOutput()
    if output_nifti_path is None:
        # 获取文件夹路径和文件名
        folder, filename = os.path.split(stl_path)
        # 移除文件扩展名（.stl）
        base_filename = os.path.splitext(filename)[0]
        # 生成新的文件名
        output_filename = f"{base_filename}_HU{HU}_nofill.nii"
        # 生成完整的输出文件路径
        output_path = os.path.join(folder, output_filename)
        output_nifti_path = output_path
    # 计算STL的质心
    # center_of_mass_filter = vtk.vtkCenterOfMass()
    # center_of_mass_filter.SetInputData(polydata)
    # center_of_mass_filter.SetUseScalarsAsWeights(False)
    # center_of_mass_filter.Update()
    # center_of_mass = center_of_mass_filter.GetCenter()

    # 体素化
    voxelizer = vtk.vtkVoxelModeller()
    voxelizer.SetInputData(polydata)
    voxelizer.SetModelBounds(polydata.GetBounds())

    # 根据新的体素尺寸计算体素网格的尺寸
    model_bounds = polydata.GetBounds()
    # breakpoint()
    sample_dimensions = [
        int((model_bounds[1] - model_bounds[0]) / voxel_size[0]),
        int((model_bounds[3] - model_bounds[2]) / voxel_size[1]),
        int((model_bounds[5] - model_bounds[4]) / voxel_size[2])
    ]
    centre_stl = 0.5 * np.array(
        [model_bounds[1] + model_bounds[0], model_bounds[3] + model_bounds[2], model_bounds[5] + model_bounds[4]])
    voxelizer.SetSampleDimensions(sample_dimensions[0], sample_dimensions[1], sample_dimensions[2])
    voxelizer.SetScalarTypeToUnsignedShort()
    voxelizer.SetMaximumDistance(0.05)
    voxelizer.SetModelBounds(model_bounds)
    voxelizer.Update()

    voxelized_data = voxelizer.GetOutput()

    # 获取体素数据作为numpy数组
    point_data = voxelized_data.GetPointData()
    if point_data is not None:
        scalars = point_data.GetScalars()
        if scalars is not None:
            voxelized_array = numpy_support.vtk_to_numpy(scalars)
            voxelized_array = voxelized_array.reshape(
                (sample_dimensions[0], sample_dimensions[1], sample_dimensions[2]),
                order='F')  # 注意这里的'F'，VTK以Fortran顺序存储数据
        else:
            raise ValueError("No scalar data in the voxelized data.")
    else:
        raise ValueError("No point data in the voxelized data.")

    # 创建NIfTI图像
    affine = np.eye(4)
    affine[:3, 3] = centre_stl - 0.5 * np.array(
        [voxel_size[0] * sample_dimensions[0], voxel_size[1] * sample_dimensions[1],
         voxel_size[2] * sample_dimensions[2]])
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]  # to be consistent with bone nif file

    img = nib.Nifti1Image(voxelized_array, affine)

    # get data
    data = img.get_fdata()
    # substitute 0 with -1000, 1 with 10000
    new_data = np.where(data <= 0, -1000, HU)
    header = img.header
    header.set_xyzt_units('mm')
    # create a new nii image
    new_img = nib.Nifti1Image(new_data, img.affine, header)

    # 保存NIfTI图像
    nib.save(new_img, output_nifti_path)
    return output_nifti_path


# 定义输入和输出路径

folder_path = r"data_files/anatomies"
folder_path = r"G:\flumatch_model"

for stl_file in os.listdir(folder_path):
    # 对每个 .stl 文件执行操作
    file_path=os.path.join(folder_path,stl_file)
    if os.path.isfile(file_path) and stl_file.lower().endswith('.stl'):

        base_filename = os.path.splitext(stl_file)[0]
        # 生成新的文件名
        output_filename = f"{base_filename}_HU{10000}_filled.nii"
        if not os.path.exists(os.path.join(folder_path,output_filename)):
            stl_path=os.path.join(folder_path,stl_file)
            print(f"Processing {stl_path}...")
            voxel_size = (0.4, 0.4, 0.4)
            HU=10000
            print(f'voxel_size={voxel_size},HU={HU}')
            # 执行转换, carefully thought about this before run it
            output_nifti_path = stl_to_nifti(stl_path, voxel_size=voxel_size,HU=HU)
            print(f'save to {output_nifti_path}')
            new_output_filename = filling_stl(output_nifti_path,HU=HU)
            print(f'save to {new_output_filename}')
            # load nifti文件
            img = nib.load(new_output_filename)

            data = img.get_fdata()
            print(data.shape)
            print(img)

breakpoint()

# compare to SUBN_02_Femur_RE_Volume
# example = nib.load('test_example_files/SUBN_02_Femur_RE_Volume.nii')
# example_data = example.get_fdata()
#
# num_voxels_3000 = np.sum(example_data > 3000)
#
# print("例子图像尺寸:", example_data.shape)
# print("值为10000的体素数量:", num_voxels_3000)
# print(example)
