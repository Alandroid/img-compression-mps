import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from utils_ND import *
from scipy.fftpack import dct, idct
import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Time to run {func.__name__}: {elapsed_time:.8f} seconds")
        return result
    return wrapper


class NDMPS:
    def __init__(self, mps=None, qubit_size=None, encoding_map=None, norm=True, mode="Std", dim = None):
        self.qubit_size = qubit_size
        self.encoding_map = encoding_map
        self.mps = mps
        self.norm = norm #Normalize matrix data
        #Compression mode 
        # "Std" standard Block Encoding
        # "DCT" discrete cosine fourier transform before compression
        self.mode = mode 
        self.dim = dim
    
    @classmethod
    # @time_function
    def from_tensor(cls, tensor, norm = False, mode = "Std"):
        qubit_size, encoding_map = gen_encoding_map(tensor.shape)
        encoding_map = np.moveaxis(encoding_map, 0, -1)

        #check for flags
        if norm:
            tensor = tensor / (np.linalg.norm(tensor))
        if mode == "DCT":
            tensor = dct(tensor, norm = "ortho")

        #initialize tensor
        contracted_tensor = np.empty(shape = tuple(qubit_size), dtype=tensor.dtype)

        # rearange the data
        k = encoding_map.shape[-1]
        flat_data = tensor.flatten()
        flat_new_indices = encoding_map.reshape(-1, k).astype(int)
        new_shape = [flat_new_indices[:, dim].max() + 1 for dim in range(k)]
        indices = tuple(flat_new_indices[:, dim] for dim in range(k))
        contracted_tensor[indices] = flat_data
        
        # Create MPS
        mps = qtn.MatrixProductState.from_dense(contracted_tensor, dims = tuple(qubit_size))

        return cls(mps, qubit_size, encoding_map, norm, mode, len(np.shape(tensor)))

    # @time_function
    def compression_ratio(self):
        initial_N = np.prod(self.qubit_size)
        compressed_N = self.number_elements_in_MPS()
        # TODO: also implement the compression rate in bits / bits
        return compressed_N / initial_N
        
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
    def continuous_compress(self, cutoff, print_ratio = True):
        compress_list = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1]) * cutoff
        for c in compress_list:
            self.compress(c)
            if print_ratio:
                print(f"Compression ratio at {c}: {self.compression_ratio()}")


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
    def to_tensor(self):
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
        k = self.encoding_map.shape[-1]
        
        recovered_tensor = np.empty(self.encoding_map.shape)
        contracted_mps = contracted_mps.data
        recovered_tensor = contracted_mps[tuple(self.encoding_map[..., dim] for dim in range(k))]
        
        if self.mode == "Std":
            return recovered_tensor
        elif self.mode == "DCT":
            return idct(recovered_tensor, norm = "ortho")
        
    def show(self):
        """
        Displays the MPS picture.
        """
        self.mps.show()

    def bond_sizes(self):
        """
        Returns the bond dimensions
        """
        return self.mps.bond_sizes()