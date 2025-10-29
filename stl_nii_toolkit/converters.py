import os
import numpy as np
import nibabel as nib
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import torch


def give_contact_num(ray):
    """计算ray中>0的voxel数量"""
    return int(np.sum(ray > 0))


def filling_stl(output_nifti_path, HU=10000, show_animation=True, use_torch=True):
    """
    加速版filling_stl，逻辑与老代码完全一致。
    仅使用NumPy或Torch加速，不改变affine或阈值逻辑。
    """
    img = nib.load(output_nifti_path)
    data = img.get_fdata().astype(np.float32)

    folder, filename = os.path.split(output_nifti_path)
    base_filename = os.path.splitext(filename)[0]
    test_image_folder = rf'data_files/anatomies/test_images/{base_filename}'
    os.makedirs(test_image_folder, exist_ok=True)

    # ✅ 可选Torch加速
    if use_torch:
        data_t = torch.tensor(data, device="cuda" if torch.cuda.is_available() else "cpu")

    nz, ny, nx = data.shape
    print(f"[INFO] Filling start: shape={data.shape}, device={'cuda' if torch.cuda.is_available() and use_torch else 'cpu'}")

    for i in range(nz):
        if use_torch:
            image = data_t[i]
        else:
            image = data[i]

        # mask：voxel<0 才需要考虑
        mask = image < 0
        if not torch.any(mask) if use_torch else not np.any(mask):
            continue

        # torch版本逻辑，替代原4向扫描
        if use_torch:
            pos_mask = image > 0
            # 在4个方向上统计每个点前/后的正voxel数
            pos_x = torch.cumsum(pos_mask, dim=1)
            pos_y = torch.cumsum(pos_mask, dim=0)
            pos_x_rev = torch.cumsum(torch.flip(pos_mask, [1]), dim=1)
            pos_y_rev = torch.cumsum(torch.flip(pos_mask, [0]), dim=0)
            pos_x_rev = torch.flip(pos_x_rev, [1])
            pos_y_rev = torch.flip(pos_y_rev, [0])

            # 同时4个方向都有>0的点 → 填充
            fill_mask = (pos_x > 0) & (pos_y > 0) & (pos_x_rev > 0) & (pos_y_rev > 0)
            image[mask & fill_mask] = HU

            data_t[i] = image

        else:
            pos_mask = image > 0
            pos_x = np.cumsum(pos_mask, axis=1)
            pos_y = np.cumsum(pos_mask, axis=0)
            pos_x_rev = np.flip(np.cumsum(np.flip(pos_mask, axis=1), axis=1), axis=1)
            pos_y_rev = np.flip(np.cumsum(np.flip(pos_mask, axis=0), axis=0), axis=0)
            fill_mask = (pos_x > 0) & (pos_y > 0) & (pos_x_rev > 0) & (pos_y_rev > 0)
            image[mask & fill_mask] = HU
            data[i] = image

        # ✅ 动画显示
        if show_animation and (i % max(1, nz // 100) == 0):
            plt.clf()
            plt.imshow(image.cpu().numpy() if use_torch else image, cmap='gray', vmin=-1000, vmax=HU)
            plt.title(f"Slice {i+1}/{nz}")
            plt.axis('off')
            plt.pause(0.05)

    # ✅ 保存
    filled_data = data_t.cpu().numpy() if use_torch else data
    new_img = nib.Nifti1Image(filled_data, img.affine, img.header)
    new_output_filename = output_nifti_path.replace("nofill", "filled")
    nib.save(new_img, new_output_filename)
    print(f"[OK] Filled NIfTI saved to: {new_output_filename}")
    return new_output_filename


def stl_to_nifti(stl_path, output_nifti_path=None, HU=10000, voxel_size=(1, 1, 1)):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    polydata = reader.GetOutput()

    if output_nifti_path is None:
        folder, filename = os.path.split(stl_path)
        base_filename = os.path.splitext(filename)[0]
        output_filename = f"{base_filename}_HU{HU}_nofill.nii"
        output_nifti_path = os.path.join(folder, output_filename)

    model_bounds = polydata.GetBounds()
    sample_dimensions = [
        int((model_bounds[1] - model_bounds[0]) / voxel_size[0]),
        int((model_bounds[3] - model_bounds[2]) / voxel_size[1]),
        int((model_bounds[5] - model_bounds[4]) / voxel_size[2])
    ]
    centre_stl = 0.5 * np.array([
        model_bounds[1] + model_bounds[0],
        model_bounds[3] + model_bounds[2],
        model_bounds[5] + model_bounds[4]
    ])

    voxelizer = vtk.vtkVoxelModeller()
    voxelizer.SetInputData(polydata)
    voxelizer.SetSampleDimensions(*sample_dimensions)
    voxelizer.SetScalarTypeToUnsignedShort()
    voxelizer.SetMaximumDistance(0.05)
    voxelizer.SetModelBounds(model_bounds)
    voxelizer.Update()

    voxelized_data = voxelizer.GetOutput()
    scalars = voxelized_data.GetPointData().GetScalars()
    voxelized_array = numpy_support.vtk_to_numpy(scalars).reshape(
        (sample_dimensions[0], sample_dimensions[1], sample_dimensions[2]),
        order='F'
    )

    affine = np.eye(4)
    affine[:3, 3] = centre_stl - 0.5 * np.array([
        voxel_size[0] * sample_dimensions[0],
        voxel_size[1] * sample_dimensions[1],
        voxel_size[2] * sample_dimensions[2]
    ])
    affine[0, 0], affine[1, 1], affine[2, 2] = voxel_size

    new_data = np.where(voxelized_array > 0, HU, -1000)
    new_img = nib.Nifti1Image(new_data.astype(np.int16), affine)
    new_img.header.set_xyzt_units('mm')
    nib.save(new_img, output_nifti_path)
    print(f"[OK] Saved NIfTI (nofill): {output_nifti_path}")
    return output_nifti_path


def process_single_stl(stl_path, voxel_size=(0.4, 0.4, 0.4), HU=10000, show_animation=True):
    if not os.path.isfile(stl_path) or not stl_path.lower().endswith('.stl'):
        raise ValueError(f"Invalid STL file path: {stl_path}")

    folder_path, stl_file = os.path.split(stl_path)
    base_filename = os.path.splitext(stl_file)[0]
    output_filename = f"{base_filename}_HU{HU}_filled.nii"
    output_path = os.path.join(folder_path, output_filename)

    if os.path.exists(output_path):
        print(f"[SKIP] File already exists: {output_path}")
        return output_path

    print(f"\n=== Processing {stl_file} ===")
    print(f"voxel_size={voxel_size}, HU={HU}")

    output_nifti_path = stl_to_nifti(stl_path, voxel_size=voxel_size, HU=HU)
    print(f"[OK] Saved intermediate: {output_nifti_path}")

    new_output_filename = filling_stl(output_nifti_path, HU=HU, show_animation=show_animation)
    print(f"[OK] Saved filled NIfTI: {new_output_filename}")

    img = nib.load(new_output_filename)
    data = img.get_fdata()
    print(f"[DONE] shape={data.shape}, affine=\n{img.affine}")
    return new_output_filename


# === 示例调用 ===
if __name__ == "__main__":
    folder_path = r"C:\Users\Public\Public Dupla\github_adrian\pytorch3d_pose_refiner\test_files"
    stl_file = "Bill_Li_Femur.stl"
    stl_path = os.path.join(folder_path, stl_file)
    # faster with larger voxel size (1.0,1.0,1.0)
    output_path = process_single_stl(stl_path, voxel_size=(0.4, 0.4, 0.4), HU=1200, show_animation=True)
    print(f"\n[FINAL] Output file saved to: {output_path}")
