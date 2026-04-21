#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#ifdef F64_PRECISION
typedef double NeuralNetworkFloat;
#else
#define F64_PRECISION 0
typedef float NeuralNetworkFloat;
#endif


// ========== CUSTOM TYPES ==========

typedef struct matrix_s {
    NeuralNetworkFloat *d_matrix_array;
    int height;
    int width;
} matrix_t;

typedef struct context_s {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulPreference_t matmul_preference;
    cublasLtMatrixTransformDesc_t matrix_transform_desc;
    void *d_workspace;
    int workspace_size;
    cudaStream_t stream;
} context_t;

typedef std::map<std::tuple<int, int, bool, int, int, bool, bool>, cublasLtMatmulAlgo_t> matmul_heuristic_cache_t;

typedef std::map<std::tuple<int, int>, std::stack<matrix_t *> > unused_matrices_t;


// ========== UTILS ==========

__device__ int get_thread_id();

__device__ int get_nb_threads();

void debug(std::string s);

void cublas(cublasStatus_t status, const std::string &source);

void cuda(cudaError_t status, const std::string &source);


// ========== CUBLASLT UTILS ==========

cublasLtHandle_t create_handle();

void free_handle(cublasLtHandle_t handle);

cublasLtMatmulDesc_t create_matmul_desc();

void free_matmul_desc(cublasLtMatmulDesc_t matmul_desc);

cublasLtMatmulPreference_t create_matmul_preference();

void free_matmul_preference(cublasLtMatmulPreference_t matmul_preference);

void *create_workspace(size_t workspace_size);

void free_workspace(void *d_workspace);

cudaStream_t create_stream();

void free_stream(cudaStream_t stream);

cublasLtMatrixLayout_t create_matrix_layout(matrix_t *matrix);

void free_matrix_layout(cublasLtMatrixLayout_t matrix_layout);

cublasLtMatrixTransformDesc_t create_matrix_transform_desc();

void free_matrix_transform_desc(cublasLtMatrixTransformDesc_t matrix_transform_desc);

context_t *create_context();

void free_context(context_t *context);


// ========== CUBLASLT AND MISCELLANEOUS ==========

bool append_unused_matrix(matrix_t *matrix);

matrix_t *get_unused_matrix(int height, int width);

matrix_t *import_matrix(NeuralNetworkFloat *matrix_array, int height, int width);

NeuralNetworkFloat *export_matrix(matrix_t *matrix, int *height, int *width);

matrix_t *create_matrix(NeuralNetworkFloat value, int height, int width);

matrix_t *clone_matrix(matrix_t *matrix);

void free_matrix(matrix_t *matrix);

matrix_t *create_empty_matrix(int height, int width);

matrix_t *get_matrix(int height, int width);

NeuralNetworkFloat get(matrix_t *matrix, int row, int column);

NeuralNetworkFloat *get_result(matrix_t *matrix, int width);

cublasLtMatrixLayout_t create_matrix_layout_from_dimension(int height, int width);

cublasLtMatmulHeuristicResult_t compute_matmul_heuristic(context_t *context, cublasLtMatrixLayout_t a_layout,
                                                         cublasLtMatrixLayout_t b_layout,
                                                         cublasLtMatrixLayout_t c_layout);

cublasLtMatmulAlgo_t get_heuristic(context_t *context, int height_lhs, int width_lhs, bool trans_lhs, int height_rhs,
                                   int width_rhs, bool trans_rhs, bool has_bias);

void set_matmul_desc_epilogue_bias(context_t *context, matrix_t *bias);

void remove_matmul_desc_epilogue_bias(context_t *context);

void set_matmul_desc_transpose(context_t *context, bool transpose_lhs, bool transpose_rhs);

void remove_matmul_desc_transpose(context_t *context);


// ========== KERNELS ==========

__global__ void apply_sigmoid_kernel(NeuralNetworkFloat *d_matrix, int len);

__global__ void apply_lhs_rhs_x_times_one_minus_x_then_times_kernel(NeuralNetworkFloat *d_lhs,
                                                                    NeuralNetworkFloat *d_rhs, int len);

void apply_sigmoid(context_t *context, matrix_t *matrix);

void apply_lhs_rhs_x_times_one_minus_x_then_times(context_t *context, matrix_t *lhs, matrix_t *rhs);


// ========== MAIN FUNCTIONS ==========

matrix_t *get_layer_output(context_t *context, matrix_t *input, matrix_t *weight, matrix_t *bias);

matrix_t *dot(context_t *context, matrix_t *lhs, bool transpose_lhs, matrix_t *rhs, bool transpose_rhs);

void apply_subtract_with_coef(context_t *context, matrix_t *lhs, matrix_t *rhs, NeuralNetworkFloat coef);


// ========== API ==========

extern "C" {
matrix_t *api_import_matrix(NeuralNetworkFloat *matrix_array, int height, int width) {
    return import_matrix(matrix_array, height, width);
}

NeuralNetworkFloat *api_export_matrix(matrix_t *matrix, int *height, int *width) {
    return export_matrix(matrix, height, width);
}

matrix_t *api_create_matrix(NeuralNetworkFloat value, int height, int width) {
    return create_matrix(value, height, width);
}

matrix_t *api_clone_matrix(matrix_t *matrix) {
    return clone_matrix(matrix);
}

void api_free_matrix(matrix_t *matrix) {
    free_matrix(matrix);
}

NeuralNetworkFloat api_get(matrix_t *matrix, int row, int column) {
    return get(matrix, row, column);
}

NeuralNetworkFloat *api_get_result(matrix_t *matrix, int width) {
    return get_result(matrix, width);
}

context_t *api_create_context() {
    return create_context();
}

void api_free_context(context_t *context) {
    free_context(context);
}

matrix_t *api_get_layer_output(context_t *context, matrix_t *input, matrix_t *weight, matrix_t *bias) {
    return get_layer_output(context, input, weight, bias);
}

matrix_t *api_dot(context_t *context, matrix_t *lhs, bool transpose_lhs, matrix_t *rhs, bool transpose_rhs) {
    return dot(context, lhs, transpose_lhs, rhs, transpose_rhs);
}

void api_apply_lhs_rhs_x_times_one_minus_x_then_times(context_t *context, matrix_t *lhs, matrix_t *rhs) {
    return apply_lhs_rhs_x_times_one_minus_x_then_times(context, lhs, rhs);
}

void api_apply_subtract_with_coef(context_t *context, matrix_t *lhs, matrix_t *rhs, NeuralNetworkFloat coef) {
    return apply_subtract_with_coef(context, lhs, rhs, coef);
}
}

#endif //CUDA_KERNEL_H
