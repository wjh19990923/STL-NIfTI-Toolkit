<p align="left">
  <img src="data/lmb_logo.png" alt="LMB Logo" width="360"/>
</p>

# STL-NIfTI Toolkit

A lightweight toolkit for converting, transforming and processing STL and NIfTI files in orthopedic research.

## Features
- 🔄 Convert STL to NII file (NIfTI) so that volumetric rendering could work better:
<div style="display: flex; gap: 10px;">
  <img src="data/test_flumatch_C_SIGM_02_st_d_02_012.tif_stl.png" width="49%"/>
  <img src="data/test_flumatch_C_SIGM_02_st_d_02_012.tif_ct.png" width="49%"/>
</div>

- 📐 find rigid transformations of STL and NIfTI models using cloud point ICP method.

<div style="display: flex; gap: 10px;">
  <img src="data/transform_match.png" width="49%"/>
  <img src="data/transform_slicer.png" width="49%"/>
</div>
- 🔻 Downsample or resample NIfTI files to target resolution to save storage

- 🦵 Compatible with femur/tibia/prosthesis pre-op & post-op data

- 📦 Lightweight and easy to use

## Installation
```bash
pip install -r requirements.txt
