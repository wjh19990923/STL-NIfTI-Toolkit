import sys
sys.path.append("..")
from stl_nii_toolkit.downsample import downsample_nifti

if __name__ == "__main__":
    niiFilePath = rf"D:\kneefit_model_nii\SUBN_02_Tibia_RE_Volume.nii"
    downsample_nifti(niiFilePath, scalingFactor=1)
    downsample_nifti(niiFilePath, scalingFactor=2)
