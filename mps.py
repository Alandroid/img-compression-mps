import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from utils import *
from scipy.fftpack import dct, idct
import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Time to run {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


class BWMPS:
    def __init__(self, mps=None, qubit_size=None, encoding_map=None, norm=True, mode="Std"):
        self.qubit_size = qubit_size
        self.encoding_map = encoding_map
        self.mps = mps
        self.norm = norm #Normalize matrix data
        #Compression mode 
        # "Std" standard Block Encoding
        # "DCT" discrete cosine fourier transform before compression
        self.mode = mode 
    
    @classmethod
    # @time_function
    def from_matrix(cls, matrix, norm = True, mode = "Std"):
        qubit_size, encoding_map = get_block_encoding_map(matrix.shape)

        #check for flags
        if norm:
            matrix = matrix / (np.linalg.norm(matrix))
        if mode == "DCT":
            matrix = dct(matrix, norm = "ortho")

        #initialize tensor
        contracted_tensor = np.empty(shape = tuple(qubit_size))

        #encode matrix data
        # start_nested_loop = time.time()
        for i in range(encoding_map.shape[0]):  
            for j in range(encoding_map.shape[1]):
                tensor_index = encoding_map[i][j]
                contracted_tensor[tensor_index] = matrix[i][j]
        # nested_loop_time = time.time() - start_nested_loop
        # print(f"Time for nested loops: {nested_loop_time:.4f} seconds")
        #put in MPS
        # start_mps_creation = time.time()
        mps = qtn.MatrixProductState.from_dense(contracted_tensor, dims = tuple(qubit_size))
        # mps_creation_time = time.time() - start_mps_creation
        # print(f"Time to create MPS from dense tensor: {mps_creation_time:.4f} seconds")
        #return class
        return cls(mps, qubit_size, encoding_map, norm, mode)

    # @time_function
    def compression_ratio(self):
        initial_N = np.prod(self.qubit_size)
        compressed_N = self.number_elements_in_MPS()
        # TODO: also implement the compression rate in bits / bits
        return compressed_N / initial_N

    # @time_function
    def matrix_to_mps(self):
        """
        Converts a matrix to a Matrix Product State (MPS) representation.
        No arguments or returns
        """
        # Get reshaped block encoded tesnor with legs of qubit size
        self.cast_matrix_to_tensor()
        # Generate MPS from reshaped tensor with l
        self.initial_mps = qtn.MatrixProductState.from_dense(self.contracted_tensor, 
                                                             dims = tuple(self.qubit_size))
        
    # @time_function
    def compress(self, cutoff):
        """
        Compresses a Matrix Product State (MPS) by cutting bonds with a relative cutoff value.
        Arguments:
            cutoff (float): The relative cutoff value to use for bond compression.
        Returns:
            None
        """
        size = len(self.mps.sites)
        for i in np.arange(1, size):
            t1 = self.mps[i-1] # Tensor 1
            t2 = self.mps[i] # Tensor 2
            # Compress bond according to percentage * bond dimension
            qtn.tensor_compress_bond(t1, t2, cutoff = cutoff, cutoff_mode = "rel") 

    # @time_function
    def number_elements_in_MPS(self):
        """
        Returns the number of tensor elements in the quimb MPS.
        Parameters:
            mps: quimb MatrixProductState object
        Returns:
            int: The total number of tensor elements in the MPS."""
        return sum(t.size for t in self.mps)
    
    # @time_function
    def mps_to_matrix(self):
        """
        Converts the compressed Matrix Product State (MPS) representation back to an image matrix.
        Arguments:
            None
        Returns:
            Compressed matrix
        """

        #conract mps
        contracted_mps = self.mps ^ ...

        #order tensor legs back
        for i in np.arange(len(contracted_mps.inds)):
            contracted_mps.moveindex("k"+str(i), i, inplace=True)
        
        #return in correct format
        if self.mode == "Std":
            img = [
            [contracted_mps.data[self.encoding_map[i,j]] \
             for j in range(self.encoding_map.shape[1])] for i in range(self.encoding_map.shape[0])
        ]
            return rescale_image(img)

        elif self.mode == "DCT":
            img = idct([
                [contracted_mps.data[self.encoding_map[i,j]] \
                for j in range(self.encoding_map.shape[1])] for i in range(self.encoding_map.shape[0])
            ], norm = "ortho")
            return rescale_image(img)
#%%

