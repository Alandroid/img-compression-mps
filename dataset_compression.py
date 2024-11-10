#%%
from mps import BWMPS
from utils import *
from PIL import Image
import os
from collections import defaultdict
# import time 
#%%
# Making the DataSet Black and White

# folder_in = r'/home/myron/QEL2024/mps/DIV2K_train_HR/'
# folder_out = r'/home/myron/QEL2024/mps/DIV2K_BW'

# for file in os.listdir(folder_in):
#     img = Image.open(os.path.join(folder_in, file))
#     img_bw = img.convert('L')
#     img_bw.save(os.path.join(folder_out, file))
#     # print(file)

# %%

def simple_resize(path_in, new_size = (256*3, 256*4)):

    mps_list = []
    bound = 1 # For debuging. To compress the whole dataset, set it to -1

    for i, filename in enumerate(os.listdir(path_in)):
        if i>=bound:
            break
        print("Working on image:",i,"!")
        image_path = os.path.join(path_in, filename)

        with Image.open(image_path) as img:
            img_array = resize_image(img, new_size)
        
        mps = BWMPS.from_matrix(img_array, norm = False ,mode = 'DCT')
        mps_list.append([mps, img_array/img_array.max()]) # Normalize pixel values for SSIM to work
    print("."*50)
    return mps_list

def padded_resize(path_in, new_size = (256*4, 256*3), pad_to = 1024):
    mps_list = []
    bound = 2

    pad_top_bottom = (pad_to - new_size[1]) // 2
    pad_left_right = (pad_to - new_size[0]) // 2

    for i, filename in enumerate(os.listdir(path_in)):
        if i>=bound:
            break
        print("Working on image:",i,"!")
        image_path = os.path.join(path_in, filename)

        with Image.open(image_path) as img:
            img_array = resize_image(img, new_size)

        img_padded = np.pad(img_array, 
                            ((pad_top_bottom, pad_top_bottom), 
                             (pad_left_right, pad_left_right)), mode='constant')
        
        mps = BWMPS.from_matrix(img_padded, norm = False ,mode = 'DCT')
        mps_list.append([mps, img_padded/img_padded.max()])

    print("."*50)
    return mps_list


#%%
path_in = r'/home/myron/QEL2024/mps/DIV2K_BW/' # CHANGE PATH TO LOAD IMAGES
mps_list = simple_resize(path_in, new_size=(512, 512)) # first value is the number of columns
                                                        # second value is the number of rows

#%%
# IF YOU WANT TO CALCULATE PSNR, 
# COMMENT OUT EVERYTHIN THAT HAS THE NAME SSIM AND 
# UNCOMMENT EVERTHING THAT HAS THE NAME PSNR

cutoff_list = np.arange(0, 4, 0.005).tolist()

compression_ratio_DCT = defaultdict(dict)

ssim_dict = defaultdict(dict)
# psnr_dict = defaultdict(dict)


for i, (mps, img_array) in enumerate(mps_list):
    print(f"Working on image: {i}!")

    compression_ratio_DCT[f'Image{i}'] = []
    
    ssim_dict[f'Image{i}'] = []
    # psnr_dict[f'Image{i}'] = []
    
    for cutoff in (cutoff_list):
        mps.compress(cutoff)
        final_matrix = mps.mps_to_matrix() 
        compression_ratio_DCT[f'Image{i}'].append(mps.compression_ratio())

        # psnr_dict[f'Image{i}'].append(output_image_quality(img_array, 
        #                                                 final_matrix, metric="psnr"))

        ssim_dict[f'Image{i}'].append(output_image_quality(img_array, 
                                                        final_matrix, metric="ssim"))
        
        if 1/mps.compression_ratio() > 50:
            print(f"Image {i} stopped at: ", 1/mps.compression_ratio())
            print('.'*50)
            break

# %%

import json
import matplotlib.pyplot as plt

# Function to save dictionaries to a txt file
def save_dicts_to_txt(file_path, psnr_dict, compression_ratio_DCT):
    with open(file_path, 'w') as file:
        data = {
            "psnr_dict": psnr_dict,
            "compression_ratio_DCT": compression_ratio_DCT
        }
        json.dump(data, file)

# Function to load dictionaries from a txt file
def load_dicts_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        psnr_dict = data["psnr_dict"]
        compression_ratio_DCT = data["compression_ratio_DCT"]
    return psnr_dict, compression_ratio_DCT

# TO LOAD DATA COMMENT OUT THE save_dicts_to_txt COMMAND 
# UNCOMMENT THE loaded_psnr_dict OR THE loaded_ssim_dict

# FILE PATH ALSO INCLUDES THE NAME OF THE FILE
file_path = "compression_ssim_data2.txt"

# save_dicts_to_txt(file_path, psnr_dict, compression_ratio_DCT)
save_dicts_to_txt(file_path, ssim_dict, compression_ratio_DCT)

# loaded_psnr_dict, loaded_compression_ratio_DCT = load_dicts_from_txt(file_path)
# loaded_ssim_dict, loaded_compression_ratio_DCT = load_dicts_from_txt(file_path)


# %%
# IN THIS PART WE CREATE THE NECESSARY PLOTS
# IF YOU ARE WORKING WITH PSNR JUST UNCOMMENT EVERTHING RELTED TO IT AND COMMENT OUT ALL THE SSIM THINGS

from scipy.interpolate import interp1d
import numpy as np

common_compression_factors = np.arange(0.645, 30.005, 0.005)

interpolated_ssim_values = []
# interpolated_psnr_values = []

for i in range(len(loaded_compression_ratio_DCT)):
    cr = np.array(loaded_compression_ratio_DCT[f'Image{i}'])

    # psnr = np.array(loaded_psnr_dict[f'Image{i}'])
    ssim_ = np.array(loaded_ssim_dict[f'Image{i}'])

    boolean_list = np.where(1/cr < 100)[0]
    cf = 1/cr[boolean_list]

    # psnr = psnr[boolean_list]
    ssim_ = ssim_[boolean_list]

    # interp_func = interp1d(cf, psnr, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_func = interp1d(cf, ssim_, kind='linear', bounds_error=False, fill_value=np.nan)

    # psnr_interp = interp_func(common_compression_factors)
    # interpolated_psnr_values.append(psnr_interp)

    ssim_interp = interp_func(common_compression_factors)
    interpolated_ssim_values.append(ssim_interp)


# interpolated_psnr_values = np.array(interpolated_psnr_values)
# mean_psnr = np.nanmean(interpolated_psnr_values, axis=0)
# std_psnr = np.nanstd(interpolated_psnr_values, axis=0)

interpolated_ssim_values = np.array(interpolated_ssim_values)
mean_ssim = np.nanmean(interpolated_ssim_values, axis=0)
std_ssim = np.nanstd(interpolated_ssim_values, axis=0)

#%%
plt.figure(figsize=(12, 6))
plt.plot(common_compression_factors, mean_ssim, label='Average ssim', color='blue')
plt.fill_between(
    common_compression_factors,
    mean_ssim - std_ssim,
    mean_ssim + std_ssim,
    color='blue',
    alpha=0.2,
    label='±1 Standard Deviation')

plt.xlabel('Compression Factor')
plt.ylabel('SSIM')
# plt.ylim(15,35)
plt.title('Average SSIM vs. Compression Factor')
plt.legend()
plt.grid(True)
plt.show()
#%%
# THIS IS TO CREATE THE HISTOGRAMS TO SEE IF THE DISTRIBUTION IS GAUSSIAN

ssim_list = []

for i in interpolated_ssim_values[:,500]:
    ssim_list.append(i)


plt.figure(figsize=(10, 6))
plt.hist(ssim_list, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('ssim')
plt.ylabel('Frequency')
plt.title('Histogram of SSIM for All Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
