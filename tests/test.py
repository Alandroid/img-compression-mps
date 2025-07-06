import sys
import os
from pathlib import Path

# from src.compression.mps_ND import NDMPS
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from copy import deepcopy

# Manually set the working directory to the project root
project_root = Path("/scratch/m/M.Theocharakis/paper/img-compression-mps")
os.chdir(project_root)

# Add the 'src' folder to sys.path so imports work
sys.path.insert(0, str(project_root / "src"))

# Now you can import your modules
from compression.utils_ND import *
from compression.mps_ND import *

mri_file = './Data/ds003799-2.0.0_sorted/ses-1/sub-season101_ses-1_acq-MPrageHiRes_T1w.nii.gz'
img = nib.load(mri_file)
img_data = img.get_fdata()
print(type(img_data))  # it's a numpy array!
print(img_data.shape)

mid_slice_x = img_data[128, :, :]
print(mid_slice_x.shape)
# Note that the transpose the slice (using the .T attribute).
# This is because imshow plots the first dimension on the y-axis and the
# second on the x-axis, but we'd like to plot the first on the x-axis and the
# second on the y-axis. Also, the origin to "lower", as the data was saved in
# "cartesian" coordinates.
mps = NDMPS.from_tensor(img_data, norm = False, mode="Std")
mps.compression_ratio()
mps.continuous_compress(0.025)
mps.compression_ratio()
mps.compression_ratio_on_disk()
recovered_img = mps.to_tensor()
calc_PSNR(img_data, recovered_img)
compute_ssim_2D(img_data, recovered_img)
avg_SSIM_3D(img_data, recovered_img)