use crate::players::neural_network::neural_networks_types::cuda::handle_pointer::HandlePointer;
use crate::players::neural_network::neural_networks_types::cuda::matrix_pointer::MatrixPointer;
use crate::players::neural_network::neural_networks_types::matrix::Matrix;
use std::ffi::c_void;

#[link(name = "kernel", kind = "static")]
unsafe extern "C" {
    fn import_matrix_array(matrix_array: *mut c_void, height: i32, width: i32) -> *mut c_void;
    fn clone_matrix(matrix: *mut c_void) -> *mut c_void;
    fn create_matrix(height: i32, width: i32, default_value: f64) -> *mut c_void;
    fn free_matrix(matrix: *mut c_void);
    fn get(matrix: *mut c_void, i: i32, j: i32) -> f64;

    fn create_handle() -> *mut c_void;
    fn free_handle(handle: *mut c_void);

    fn plus(handle: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn scale(handle: *mut c_void, alpha: f64, m: *mut c_void) -> *mut c_void;
    fn dot(
        handle: *mut c_void,
        a: *mut c_void,
        t_a: bool,
        b: *mut c_void,
        t_b: bool,
    ) -> *mut c_void;
    fn subtract_with_coef(
        handle: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        coef: f64,
    ) -> *mut c_void;
    fn sigmoid(m: *mut c_void) -> *mut c_void;
    fn x_times_one_minus_x(m: *mut c_void) -> *mut c_void;
    fn times(a: *mut c_void, b: *mut c_void) -> *mut c_void;
}

pub fn cuda_import_matrix(matrix: &Matrix) -> MatrixPointer {
    // Column-major ordering is required by cuBLAS
    let mut matrix_array = Vec::new();
    for j in 0..matrix.width() {
        for i in 0..matrix.height() {
            matrix_array.push(matrix.get(i, j));
        }
    }
    unsafe {
        MatrixPointer::from(import_matrix_array(
            matrix_array.as_mut_ptr() as *mut c_void,
            matrix.height() as i32,
            matrix.width() as i32,
        ))
    }
}

pub fn cuda_clone_matrix(matrix: &MatrixPointer) -> MatrixPointer {
    unsafe { MatrixPointer::from(clone_matrix(matrix.pointer)) }
}

pub fn cuda_create_matrix(height: usize, width: usize, default_value: f64) -> MatrixPointer {
    unsafe { MatrixPointer::from(create_matrix(height as i32, width as i32, default_value)) }
}

pub fn cuda_free_matrix(matrix: &mut MatrixPointer) {
    unsafe {
        free_matrix(matrix.pointer);
    }
}

pub fn cuda_get(matrix: &MatrixPointer, i: usize, j: usize) -> f64 {
    unsafe { get(matrix.pointer, i as i32, j as i32) }
}

pub fn cuda_create_handle() -> HandlePointer {
    unsafe { HandlePointer::from(create_handle()) }
}

pub fn cuda_free_handle(handle: &mut HandlePointer) {
    unsafe {
        free_handle(handle.pointer);
    }
}

pub fn cuda_plus(handle: &HandlePointer, a: &MatrixPointer, b: &MatrixPointer) -> MatrixPointer {
    unsafe { MatrixPointer::from(plus(handle.pointer, a.pointer, b.pointer)) }
}

pub fn cuda_scale(handle: &HandlePointer, alpha: f64, m: &MatrixPointer) -> MatrixPointer {
    unsafe { MatrixPointer::from(scale(handle.pointer, alpha, m.pointer)) }
}

pub fn cuda_dot(
    handle: &HandlePointer,
    a: &MatrixPointer,
    t_a: bool,
    b: &MatrixPointer,
    t_b: bool,
) -> MatrixPointer {
    unsafe { MatrixPointer::from(dot(handle.pointer, a.pointer, t_a, b.pointer, t_b)) }
}

pub fn cuda_subtract_with_coef(
    handle: &HandlePointer,
    a: &MatrixPointer,
    b: &MatrixPointer,
    coef: f64,
) -> MatrixPointer {
    unsafe {
        MatrixPointer::from(subtract_with_coef(
            handle.pointer,
            a.pointer,
            b.pointer,
            coef,
        ))
    }
}

pub fn cuda_sigmoid(m: &MatrixPointer) -> MatrixPointer {
    unsafe { MatrixPointer::from(sigmoid(m.pointer)) }
}

pub fn cuda_x_times_one_minus_x(m: &MatrixPointer) -> MatrixPointer {
    unsafe { MatrixPointer::from(x_times_one_minus_x(m.pointer)) }
}

pub fn cuda_times(a: &MatrixPointer, b: MatrixPointer) -> MatrixPointer {
    unsafe { MatrixPointer::from(times(a.pointer, b.pointer)) }
}
