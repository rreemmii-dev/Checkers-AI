use crate::players::neural_network::neural_networks_types::cuda::cuda::{
    cuda_clone_matrix, cuda_free_matrix,
};
use std::ffi::c_void;
use std::ptr::null_mut;

pub struct MatrixPointer {
    pub pointer: *mut c_void,
}

impl MatrixPointer {
    pub fn null() -> MatrixPointer {
        MatrixPointer::from(null_mut())
    }
}

impl From<*mut c_void> for MatrixPointer {
    fn from(pointer: *mut c_void) -> Self {
        MatrixPointer { pointer }
    }
}

impl Clone for MatrixPointer {
    fn clone(&self) -> Self {
        MatrixPointer::from(cuda_clone_matrix(self))
    }
}

impl Drop for MatrixPointer {
    fn drop(&mut self) {
        cuda_free_matrix(self)
    }
}
