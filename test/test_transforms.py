import sys
sys.path.append("..")

from stl_nii_toolkit.transforms import find_transform, apply_transform



if __name__ == "__main__":
    # femur
    # # 配准输入
    target_path = r"D:\Toni\DPHK_01_Femur_notshifted_FC.stl"
    source_path = r"D:\Toni\femur_r.stl"

    # # 应用变换后保存路径
    output_path = r"D:\Toni\femur_r_transformed.stl"

    # # 配准并获得变换
    T_final = find_transform(source_path, target_path, visualize=True)

    print("Final Transformation Matrix:\n", T_final)
    # # 应用变换并保存
    apply_transform(source_path, T_final, output_path)
    
