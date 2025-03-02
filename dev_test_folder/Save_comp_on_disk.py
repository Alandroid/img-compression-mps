#%%
import sys
import os
from pathlib import Path

# Automatically find the project root (assumes "src" is in the project)
current_file = Path(__file__).resolve() if '__file__' in locals() else Path.cwd()
project_root = current_file
while not (project_root / "src").exists() and project_root != project_root.parent:
    project_root = project_root.parent

# Ensure we found the correct project root
if not (project_root / "src").exists():
    raise FileNotFoundError("Could not find project root containing 'src' directory.")

# Set the working directory to the project root
os.chdir(project_root)
from src.compression.mps_ND import NDMPS
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from src.compression.utils_ND import *
import pickle as pkl
from copy import deepcopy
#%%
mri_file = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/MRI Dataset/sub-01_T1w.nii.gz'
img = nib.load(mri_file)
img_data = img.get_fdata()
print(type(img_data))  # it's a numpy array!
print(img_data.shape)
#%%

mid_slice_x = img_data[80, :, :]
print(mid_slice_x.shape)
# Note that the transpose the slice (using the .T attribute).
# This is because imshow plots the first dimension on the y-axis and the
# second on the x-axis, but we'd like to plot the first on the x-axis and the
# second on the y-axis. Also, the origin to "lower", as the data was saved in
# "cartesian" coordinates.
plt.imshow(mid_slice_x.T, cmap='gray', origin='lower')
plt.xlabel('First axis')
plt.ylabel('Second axis')
plt.colorbar(label='Signal intensity')
plt.show()

# %%
mps = NDMPS.from_tensor(img_data, norm = False)
mps_og = deepcopy(mps)
mps.compression_ratio()
# %%
mps.continuous_compress(0.05)
mps.compression_ratio()
#%%
mps_og.continuous_compress(0.05)
mps_og.compression_ratio()

# %%
recovered_img = mps.to_tensor()
plt.imshow(recovered_img[80,:,:].T, cmap='gray', origin='lower')
plt.xlabel('First axis')
plt.ylabel('Second axis')
plt.colorbar(label='Signal intensity')
plt.show()
#TODO: Put into benchmark the compressed disk size in json, put into the number of elements, compressed elemtnst
tensor_list_og = mps.return_tensors_data()
# %%

print("Compressed Storage space: ", mps.get_bytesize_on_disk())
print("Uncompressed Storage space on disk: ", mps.get_storage_space(np.uint16))
print("compression ratio on disk:", mps.compression_ratio_on_disk(np.uint16))
print("compression ratio: ", mps.compression_ratio())
print(mps.number_elements_in_MPS())
# %%

tensor_int = mps.compress_to_dtype(np.uint16, replace=True)
#%%
recov_img_int = mps.to_tensor()
plt.imshow(recov_img_int[80,:,:].T, cmap='gray', origin='lower')
#%%
print("SSIM uncom & int_quant: ",compute_ssim_2D(img_data[80,:,:], recov_img_int[80,:,:]))
print("SSIM uncom & comp: ",compute_ssim_2D(img_data[80,:,:], recovered_img[80,:,:]))

mps.get_storage_space(np.uint16)
# %%
mps.replace_tensordata(tensor_list_og)
recov_img_int = mps.to_tensor()
#%%
recov_img_og = mps.to_tensor()
compute_ssim_2D(img_data[80,:,:], recov_img_og[80,:,:])
#%%
#print(tensor_list_og)
print(mps.return_tensors_data())

# %%
print(img.header["bitpix"])

# %%
import gzip
import io
#%%

original_bytelength = 0
compressed_bytelength = 0

for i in np.arange(len(tensor_int)):
    array_bytes = tensor_int[i].tobytes()
    original_bytelength += len(array_bytes)
    buffer = io.BytesIO()
    print("original length:", len(array_bytes))
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
        gz_file.write(array_bytes)
    compressed_data = buffer.getvalue()
    compressed_size = len(compressed_data)
    compressed_bytelength += compressed_size
    print("Compressed size in bytes:", compressed_size)

# %%
array_bytes = tensor_int[3].tobytes()
buffer = io.BytesIO()
with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
    gz_file.write(array_bytes)

# Get the compressed data as bytes and its size
compressed_data = buffer.getvalue()
compressed_size = len(compressed_data)

print("Compressed size in bytes:", compressed_size)
# %%
original_bytelength
# %%

compressed_bytelength/original_bytelength
# %%
mps.get_storage_space(np.uint16)
# %%
mps.number_elements_in_MPS() *2
# %%
