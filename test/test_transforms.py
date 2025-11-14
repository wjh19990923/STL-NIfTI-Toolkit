import sys
sys.path.append("..")

from stl_nii_toolkit.transforms import find_transform, apply_transform



if __name__ == "__main__":
    # # 配准输入
    # source_path = r"D:\Toni\DPHK_01_Femur_notshifted_FC.stl"
    # target_path = r"D:\Toni\femur_r.stl"

    # # 应用变换后保存路径
    # output_path = r"D:\Toni\DPHK_01_Femur_notshifted_FC_transformed.stl"

    # # 配准并获得变换
    # T_final = find_transform(source_path, target_path, visualize=True)

    # print("Final Transformation Matrix:\n", T_final)
    # # 应用变换并保存
    # apply_transform(source_path, T_final, output_path)

    # 配准输入
    source_path = r"D:\Toni\DPHK_01_Tibia_notshifted_IM.stl"
    target_path = r"D:\Toni\tibia_r.stl"

    # 应用变换后保存路径
    output_path = r"D:\Toni\DPHK_01_Tibia_notshifted_IM_transformed.stl"

    # 配准并获得变换
    T_final = find_transform(source_path, target_path, visualize=True)

    print("Final Transformation Matrix:\n", T_final)
    # 应用变换并保存
    apply_transform(source_path, T_final, output_path)

