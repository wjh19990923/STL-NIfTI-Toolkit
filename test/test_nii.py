import sys 
sys.path.append("..") 
from stl_nii_toolkit.utils import nii_check

if __name__ == "__main__":
    nii_check(corresponding_nii_path=rf"T:\MITK\SUBN_02_Pattela_RE_Volume.nii")
