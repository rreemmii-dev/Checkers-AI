use crate::consts::NeuralNetworkFloat;
use crate::neural_network::types::cuda::context_pointer::ContextPointer;
use crate::neural_network::types::cuda::matrix_pointer::MatrixPointer;
use crate::neural_network::types::matrix::Matrix;
use std::ffi::c_void;

#[link(name = "kernel", kind = "static")]
unsafe extern "C" {
    fn api_import_matrix(
        matrix_array: *mut NeuralNetworkFloat,
        height: i32,
        width: i32,
    ) -> *mut c_void;
    fn api_export_matrix(
        matrix: *mut c_void,
        height: *mut i32,
        width: *mut i32,
    ) -> *mut NeuralNetworkFloat;
    fn api_create_matrix(default_value: NeuralNetworkFloat, height: i32, width: i32)
    -> *mut c_void;
    fn api_clone_matrix(matrix: *mut c_void) -> *mut c_void;
    fn api_free_matrix(matrix: *mut c_void);
    fn api_get(matrix: *mut c_void, row: i32, column: i32) -> NeuralNetworkFloat;
    fn api_get_result(matrix: *mut c_void, width: i32) -> *mut NeuralNetworkFloat;
    fn api_create_context() -> *mut c_void;
    fn api_free_context(context: *mut c_void);
    fn api_get_layer_output(
        context: *mut c_void,
        input: *mut c_void,
        weight: *mut c_void,
        bias: *mut c_void,
    ) -> *mut c_void;
    fn api_dot(
        context: *mut c_void,
        lhs: *mut c_void,
        transpose_lhs: bool,
        rhs: *mut c_void,
        transpose_rhs: bool,
    ) -> *mut c_void;
    fn api_apply_lhs_rhs_x_times_one_minus_x_then_times(
        context: *mut c_void,
        lhs: *mut c_void,
        rhs: *mut c_void,
    );
    fn api_apply_subtract_with_coef(
        context: *mut c_void,
        lhs: *mut c_void,
        rhs: *mut c_void,
        coef: NeuralNetworkFloat,
    );
}

pub fn cuda_import_matrix(matrix: &Matrix) -> MatrixPointer {
    // Column-major ordering is required by cuBLAS
    let mut matrix_array = Vec::new();
    for j in 0..matrix.width() {
        for i in 0..matrix.height() {
            matrix_array.push(matrix.get(i, j));
        }
    }
    MatrixPointer::from(unsafe {
        api_import_matrix(
            matrix_array.as_mut_ptr() as *mut NeuralNetworkFloat,
            matrix.height() as i32,
            matrix.width() as i32,
        )
    })
}

pub fn cuda_export_matrix(matrix: &MatrixPointer) -> Matrix {
    let mut height = -1;
    let mut width = -1;
    let c_matrix_array =
        unsafe { api_export_matrix(matrix.pointer, &raw mut height, &raw mut width) };
    let height = height as usize;
    let width = width as usize;
    let len = height * width;
    let matrix_array =
        unsafe { Vec::<NeuralNetworkFloat>::from_raw_parts(c_matrix_array, len, len) };

    let mut matrix = Matrix::new(height, width);

    // Column-major ordering is required by cuBLAS
    for j in 0..width {
        for i in 0..height {
            matrix.set(i, j, matrix_array[j * height + i]);
        }
    }

    matrix
}

pub fn cuda_create_matrix(
    default_value: NeuralNetworkFloat,
    height: usize,
    width: usize,
) -> MatrixPointer {
    MatrixPointer::from(unsafe { api_create_matrix(default_value, height as i32, width as i32) })
}

pub fn cuda_clone_matrix(matrix: &MatrixPointer) -> MatrixPointer {
    MatrixPointer::from(unsafe { api_clone_matrix(matrix.pointer) })
}

pub fn cuda_free_matrix(matrix: &mut MatrixPointer) {
    unsafe { api_free_matrix(matrix.pointer) }
}

pub fn cuda_get(matrix: &MatrixPointer, row: usize, column: usize) -> NeuralNetworkFloat {
    unsafe { api_get(matrix.pointer, row as i32, column as i32) }
}

pub fn cuda_get_result(matrix: &MatrixPointer, matrix_width: usize) -> Vec<NeuralNetworkFloat> {
    unsafe {
        Vec::<NeuralNetworkFloat>::from_raw_parts(
            api_get_result(matrix.pointer, matrix_width as i32),
            matrix_width,
            matrix_width,
        )
    }
}

pub fn cuda_create_context() -> ContextPointer {
    ContextPointer::from(unsafe { api_create_context() })
}

pub fn cuda_free_context(context: &mut ContextPointer) {
    unsafe { api_free_context(context.pointer) }
}

pub fn cuda_get_layer_output(
    context: &ContextPointer,
    input: &MatrixPointer,
    weight: &MatrixPointer,
    bias: &MatrixPointer,
) -> MatrixPointer {
    MatrixPointer::from(unsafe {
        api_get_layer_output(context.pointer, input.pointer, weight.pointer, bias.pointer)
    })
}

pub fn cuda_dot(
    context: &ContextPointer,
    lhs: &MatrixPointer,
    transpose_lhs: bool,
    rhs: &MatrixPointer,
    transpose_rhs: bool,
) -> MatrixPointer {
    MatrixPointer::from(unsafe {
        api_dot(
            context.pointer,
            lhs.pointer,
            transpose_lhs,
            rhs.pointer,
            transpose_rhs,
        )
    })
}

pub fn cuda_apply_lhs_rhs_x_times_one_minus_x_then_times(
    context: &ContextPointer,
    lhs: &mut MatrixPointer,
    rhs: &MatrixPointer,
) {
    unsafe {
        api_apply_lhs_rhs_x_times_one_minus_x_then_times(context.pointer, lhs.pointer, rhs.pointer);
    }
}

pub fn cuda_apply_subtract_with_coef(
    context: &ContextPointer,
    lhs: &mut MatrixPointer,
    rhs: &MatrixPointer,
    coef: NeuralNetworkFloat,
) {
    unsafe { api_apply_subtract_with_coef(context.pointer, lhs.pointer, rhs.pointer, coef) }
}
