import numpy as np
from sympy import factorint
from skimage.metrics import structural_similarity as ssim
from math import sqrt, log10
from PIL import Image
import time
#import pillow_heif

encoding_map_cache = {}

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Time to run {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

# @time_function
def multi_kronecker_product(matrices):
    """
    Computes the Kronecker product of a list of 2D integer arrays with elementwise tuple addition.
    Arguments:
    matrices (list of np.ndarray): A list of 2D numpy arrays containing integers.
    Returns:
    np.ndarray: A 2D numpy array where each element is a tuple representing the Kronecker product
                of the input matrices with elementwise tuple addition.
    """
    # Initialize result with the first matrix, converting elements to tuples
    result = np.empty(matrices[0].shape, dtype=object)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = (matrices[0][i, j],)
    
    for A in matrices[1:]:
        # Initialize the new result array with the correct shape
        new_shape = (result.shape[0] * A.shape[0], result.shape[1] * A.shape[1])
        new_result = np.empty(new_shape, dtype=object)
        
        # Perform the Kronecker product with elementwise tuple addition
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                for k in range(A.shape[0]):
                    for l in range(A.shape[1]):
                        new_result[i * A.shape[0] + k, j * A.shape[1] + l] = result[i, j] + (A[k, l],)
        
        result = new_result
    
    return result

# @time_function
def get_block_encoding_map(shape):
    """
    Generates block Encoding mapping from image data shape (m,n)
    Arguments:
        shape : (tuple): A tuple of integers representing the shape of the image data.
    Returns:
        qubit_size : (list): A list of integers representing the size of the qubits 
        for the physical legs of the MPS tensors
        kron : Mapping matrix containing, at each image pixel, tuples storing the physical qubit indices 
    """

    if shape in encoding_map_cache:
        return encoding_map_cache[shape]

    factor_dict_row = factorint(shape[0])
    factor_dict_column = factorint(shape[1])
    # Extract prime factors and repeat them according to their exponents

    # Generate ascending list of prime factors of image rows
    prime_factors_row = []
    for prime, exponent in factor_dict_row.items():
        prime_factors_row.extend([prime] * exponent)

    # Generate ascending list of prime factors of image columns
    prime_factors_column = []
    for prime, exponent in factor_dict_column.items():
        prime_factors_column.extend([prime] * exponent)

    new_rows = []
    new_columns = []
    
    # Rows and columns dont have the same number of prime factors, so we need to group them
    # Here the smallest prime factos are grouped until they rows and columnds have the same 
    # number of prime factors
    if len(prime_factors_row) > len(prime_factors_column):
        d = len(prime_factors_row) - len(prime_factors_column)
        new_columns = np.flip(np.sort(prime_factors_column))
        for i in np.arange(0,2*d,2):
            new_rows.append(prime_factors_row[i]*prime_factors_row[i+1])
        new_rows = np.sort(new_rows + prime_factors_row[2*d:])
    else:
        d = len(prime_factors_column) - len(prime_factors_row)
        new_rows = np.sort(prime_factors_row)
        for i in np.arange(0,2*d,2):
            new_columns.append(prime_factors_column[i]*prime_factors_column[i+1])
        new_columns = np.flip(np.sort(new_columns + prime_factors_column[2*d:]))
    
    # Multipy same new list of grouped prime factors for row and coulmnds together to
    # obtain the qubit size of the Block encoding
    qubit_size = new_rows* new_columns

    tensors = []
    for i in np.arange(len(qubit_size)):
        # generates Adressing tensors for Block encoding
        # tensors have flattened lenth of qubit size but are reshaped into the rows and columnds
        # prime factors of the image to fit the image size
        tensors.append(np.arange(qubit_size[i]).reshape(new_rows[i],new_columns[i]))
    kron = multi_kronecker_product(tensors)

    encoding_map_cache[shape] = (qubit_size, kron)

    return qubit_size, kron 

def resize_image(image, new_size):
    """
    Resize the input image to the given size.
    Arguments:
        image (PIL Image): The image to be resized.
        new_size (tuple): The target size (width, height).
    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    return np.array(image.resize(new_size, Image.LANCZOS))
    
# @time_function
def order_tensor_legs(tensor):
    """
    Orders the physical legs of the tensor back to its initial order.

    This function takes a quimb tensor `t` and reorders its physical legs back to 
    their initial order.
    It does this by iterating over the tags of the tensor legs and 
    moving each index to its corresponding initial position, which is k0, k1, k2, ... .
    The function modifies the tensor in place.
    Arguments:
        t (quimb tensor): The tensor to be reordered.
    Returns:
        tensor (quimb tensor): Same tensor but with ordered legs
    """
    size = len(tensor.inds)
    for i in np.arange(size):
        tensor.moveindex("k"+str(i), i, inplace=True)
    
    return tensor

# @time_function
def output_image_quality(input_matrix, output_matrix, metric='ssim'):
    if (metric == 'rmse'):
        result = compute_rmse(input_matrix, output_matrix)

    elif(metric == 'psnr'):
        result = compute_psnr(input_matrix, output_matrix)
    
    elif(metric == 'ssim'):
        result = compute_ssim(input_matrix, output_matrix)

    else:
        print('There is no such metric included// returning 0')
        return 0 
    
    return result



    # match metric:
    #     case 'rmse':
    #         result = compute_rmse(input_matrix, output_matrix)
    #     case 'psnr':
    #         result = compute_psnr(input_matrix, output_matrix)
    #     case 'ssim':
    #         result = compute_ssim(input_matrix, output_matrix)
    # return result

# @time_function
def compute_rmse(input_matrix, output_matrix):
    height = len(input_matrix)
    width = len(input_matrix[0])
    square_diff_list = [
        (input_matrix[m][n] - output_matrix[m][n])**2 \
            for m in range(height) for n in range(width)
    ]
    return sum(square_diff_list) /(width*height)


# TODO: change this for RGB plots
# @time_function
def compute_psnr(input_matrix, output_matrix):
    mse = compute_rmse(input_matrix, output_matrix)
    if(mse == 0):  
        # MSE is zero means no noise is present in the signal . 
        # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 


    # Compute Mean Squared Error (MSE)
    # mse = np.mean((np.array(input_matrix) - np.array(output_matrix)) ** 2)
    
    # if mse == 0:  
    #     # MSE is zero means the images are identical
    #     return 100  # Perfect PSNR
    
    # max_pixel = 255.0
    # psnr = 10 * log10((max_pixel ** 2) / mse)  # Correct PSNR formula
    # return psnr

# @time_function
def compute_ssim(input_matrix, output_matrix):
    # Assuming input and output images the same size
    input_array = np.array(input_matrix)
    output_array = np.array(output_matrix)
    return ssim(input_array, output_array, data_range=input_array.max() - input_array.min())


def rescale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    rescaled_image = (image - min_val)/(max_val -min_val)
    return rescaled_image

# TODO: should we use MPS fidelity for something in the future?
"""def plot_fidelity_compression(self, N=20):
        
        Plots the fidelity and compression factor against the cutoff value for a given MPS.
        Arguments:
            N: The number of points to generate in the cutoff range.
        Returns:
            None
        initial_mps = self.initial_mps.copy()
        cutoff = np.linspace(0,1,N)
        compression_ratio = []
        fidelity = np.zeros(len(cutoff))
        N_Matrix = np.prod(self.qubit_size)
        for i in np.arange(len(cutoff)):
            com_mps = self.initial_mps.copy()
            com_mps.compress(cutoff[i])
            compression_ratio.append(self.number_elements_in_MPS() / N_Matrix)
            fidelity[i] = self.initial_mps @ self.compressed_mps
            
        plt.figure()
        plt.xlabel('cutoff')
        plt.plot(cutoff, fidelity, label = 'fidelity')
        plt.plot(cutoff, compression_ratio, label = 'compression ratio')
        plt.legend()
        plt.grid()"""