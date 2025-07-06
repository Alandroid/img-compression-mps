import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from compression.utils_ND import *
import gzip
import io
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
    def __init__(self, mps: qtn.MatrixProductState = None, qubit_size: tuple = None, encoding_map: np.ndarray = None, 
                 boundary_list: list = None,norm: bool = True, norm_value = None,mode: str = "Std", dim: int = None):
        """
        Initializes the NDMPS class.
        
        Args:
            mps (qtn.MatrixProductState, optional): MPS representation.
            qubit_size (tuple, optional): Size of the qubit encoding.
            encoding_map (np.ndarray, optional): Encoding map.
            norm (bool, optional): Whether to normalize data. Defaults to True.
            mode (str, optional): Compression mode, "Std" or "DCT". Defaults to "Std".
            encoding_scheme (str, optional): Encoding method: "rowmajor", "snake", or "hierarchical". Defaults to "hierarchical".
            dim (int, optional): Dimensionality of input tensor.
        """
        self.qubit_size = qubit_size
        self.encoding_map = encoding_map
        self.mps = mps
        self.dim = dim
        self.norm = norm
        self.norm_value = norm_value
        self.mode = mode
        self.boundary_list = np.array(boundary_list) # contains the maximum and minimum value of each mps tensor
    
    @classmethod
    # @time_function
    def from_tensor(cls, tensor: np.ndarray, norm: bool = False, mode: str = "Std") -> "NDMPS":
        """
        Creates an NDMPS instance from a tensor with specified encoding scheme.
        
        Args:
            tensor (np.ndarray): Input tensor.
            norm (bool, optional): Whether to normalize the tensor. Defaults to False.
            mode (str, optional): Compression mode, "Std" or "DCT". Defaults to "Std".
            encoding_scheme (str, optional): Encoding method: "rowmajor", "snake", or "hierarchical". Defaults to "hierarchical".
        
        Returns:
            NDMPS: Instance of NDMPS class.
        """
        tensor = tensor.astype(np.float64)  # Ensure tensor is in float64 format
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
        boundary_list = []
        for arr in mps.arrays:
            boundary_list.append([np.min(arr), np.max(arr)])
        
        norm_value = np.sqrt(mps @ mps)

        return cls(mps, qubit_size, encoding_map, boundary_list, norm, norm_value, mode, len(np.shape(tensor)))


    def update_boundary_list(self):
        boundary_list = []
        for arr in self.mps.arrays:
            boundary_list.append([np.min(arr), np.max(arr)])
        self.boundary_list = np.array(boundary_list)

    # @time_function
    def compression_ratio(self):
        initial_N = np.prod(self.qubit_size)
        compressed_N = self.number_elements_in_MPS()
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
        self.update_boundary_list()
        self.update_norm()
    
    def update_norm(self):
        """
        Updates the norm value of the MPS.
        """
        self.norm_value = np.sqrt(self.mps @ self.mps)

    def continuous_compress(self, cutoff: float, print_ratio: bool = True):
        """
        Performs continuous compression at different cutoff levels.
        
        Args:
            cutoff (float): Base cutoff value.
            print_ratio (bool, optional): Whether to print compression ratio. Defaults to True.
        """
        compress_list = np.linspace(0,1,20) * cutoff
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
    
    def replace_tensordata(self, tensorlist):
        """
        Replaces the Tensor Data in the MPS tensors with the tensors in the tensorlist
        """
        for i in range(len(self.mps.arrays)):
            assert self.mps.arrays[i].shape == tensorlist[i].shape #check whether they have the same shape
            self.mps.arrays[i][:] = tensorlist[i] #replace quimb tensors with the new tensors
        self.update_boundary_list()
        self.update_norm() #update the norm value of the MPS

    def return_tensors_data(self):
        """
        Returns the data of the tensors in the MPS
        """
        return [t for t in self.mps.arrays]

    def get_bytesize_on_disk(self, dtype=np.uint16, replace = False):
        """
        Returns the bytesize of the MPS on disk with gzip compression
        If this function is called also integer truncation to this datatype is performed!!!
        Ideallly the same bytesize as the original tensor should be used
        """

        tensor_int_list = self.compress_to_dtype(dtype, replace) # Compresses the MPS tensors to a specific dtype if replace is true
        compressed_bytesize = 0
        for i in np.arange(len(tensor_int_list)):
            array_bytes = tensor_int_list[i].tobytes()
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
                gz_file.write(array_bytes)
            compressed_data = buffer.getvalue()
            compressed_size = len(compressed_data)
            compressed_bytesize += compressed_size
        return compressed_bytesize # Returns the bytesize of the MPS on disk with gzip compression
    

    def compression_ratio_on_disk(self, dtype=np.uint16, replace = False):
        """
        Returns the compression ratio on disk
        When this funciton is called also integer truncation is performed!!!
        """
        initial_N = np.prod(self.qubit_size) * get_num_bits(dtype) / 8
        compressed_N = self.get_bytesize_on_disk(dtype, replace)
        return compressed_N / initial_N
        


    def compress_to_dtype(self, dtype=np.uint16, replace = False):
        """
        Compresses the MPS tensors to a specific dtype
        For example replacing float64 numbers with uint16 and performing integer truncation

        dtype: datatype to be used for Integer truncatition
        replace: If True the tensors in the MPS are replaced with the Integer truncated tensors
        """
        tensor_int_list = [scale_to_dtype(t, dtype) for t in self.mps.arrays]
        scaled_back_tensors = [scale_back(t, b[0], b[1], dtype) for t, b in zip(tensor_int_list, self.boundary_list)]
        if replace:
            self.replace_tensordata(scaled_back_tensors)
        return tensor_int_list

    def get_storage_space(self, dtype=np.uint16, p = False):
        """
        Returns the storage space in bytes on the disk without gzip compression
        dtype: datatype to be used for calculating the storage space on the disk
        """

        storage_space = self.number_elements_in_MPS() * get_num_bits(dtype) / 8
        if p:
            print("The storage space is approximately: ", storage_space/1024, "KB")
        return storage_space
