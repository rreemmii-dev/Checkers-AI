use crate::neural_network::types::cuda::cuda::cuda_free_context;
use std::ffi::c_void;

pub struct ContextPointer {
    pub pointer: *mut c_void,
}

impl From<*mut c_void> for ContextPointer {
    fn from(pointer: *mut c_void) -> Self {
        ContextPointer { pointer }
    }
}

impl Drop for ContextPointer {
    fn drop(&mut self) {
        cuda_free_context(self);
    }
}
