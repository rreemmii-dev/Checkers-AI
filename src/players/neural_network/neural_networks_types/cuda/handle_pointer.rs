use crate::players::neural_network::neural_networks_types::cuda::cuda::cuda_free_handle;
use std::ffi::c_void;

pub struct HandlePointer {
    pub pointer: *mut c_void,
}

impl From<*mut c_void> for HandlePointer {
    fn from(pointer: *mut c_void) -> Self {
        HandlePointer { pointer }
    }
}

impl Drop for HandlePointer {
    fn drop(&mut self) {
        cuda_free_handle(self)
    }
}
