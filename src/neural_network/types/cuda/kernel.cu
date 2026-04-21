#include <cassert>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <mutex>
#include <nvtx3/nvToolsExt.h>
#include <stack>

#include "kernel.h"

#define DEBUG 0
#define MAX_UNUSED_MATRICES_SAVED 10
#define NB_THREADS_PER_BLOCK 128
#define NB_BLOCS 32

// NeuralNetworkFloat is defined in kernel.h, using F64_PRECISION compile definition

constexpr cublasDataType_t DATA_TYPE = F64_PRECISION ? CUDA_R_64F : CUDA_R_32F;
constexpr cublasComputeType_t COMPUTE_TYPE = F64_PRECISION ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
constexpr cublasDataType_t SCALE_TYPE = DATA_TYPE;
constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB

constexpr NeuralNetworkFloat ONE = 1.;
constexpr NeuralNetworkFloat ZERO = 0.;

using std::lock_guard;
using std::map;
using std::mutex;
using std::stack;
using std::string;
using std::tuple;


// ========== UTILS ==========

__device__ int get_thread_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_nb_threads() {
    return gridDim.x * blockDim.x;
}

void debug(string s) {
    if constexpr (DEBUG) {
        printf("%s\n", s.c_str());
    }
}

void cublas(cublasStatus_t status, const string &source) {
    debug(source);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CuBLAS Error: %d: %s - %s\n", status, cublasLtGetStatusName(status), cublasLtGetStatusString(status));
        printf("From: %s\n", source.c_str());
    }
    assert(status == CUBLAS_STATUS_SUCCESS);
}

void cuda(cudaError_t status, const string &source) {
    debug(source);
    if (status != cudaSuccess) {
        printf("CUDA Error: %d: %s - %s\n", status, cudaGetErrorName(status), cudaGetErrorString(status));
        printf("From: %s\n", source.c_str());
    }
    assert(status == cudaSuccess);
}


// ========== CUBLASLT UTILS ==========

cublasLtHandle_t create_handle() {
    nvtxRangePush("create_handle");
    cublasLtHandle_t handle;
    cublas(cublasLtCreate(&handle), "create_handle");
    nvtxRangePop();
    return handle;
}

void free_handle(cublasLtHandle_t handle) {
    nvtxRangePush("free_handle");
    cublas(cublasLtDestroy(handle), "free_handle");
    nvtxRangePop();
}

cublasLtMatmulDesc_t create_matmul_desc() {
    nvtxRangePush("create_matmul_desc");
    cublasLtMatmulDesc_t matmul_desc;
    cublas(cublasLtMatmulDescCreate(&matmul_desc, COMPUTE_TYPE, SCALE_TYPE), "create_matmul_desc");
    nvtxRangePop();
    return matmul_desc;
}

void free_matmul_desc(cublasLtMatmulDesc_t matmul_desc) {
    nvtxRangePush("free_matmul_desc");
    cublas(cublasLtMatmulDescDestroy(matmul_desc), "free_matmul_desc");
    nvtxRangePop();
}

cublasLtMatmulPreference_t create_matmul_preference() {
    nvtxRangePush("create_matmul_preference");
    cublasLtMatmulPreference_t matmul_preference;
    cublas(cublasLtMatmulPreferenceCreate(&matmul_preference), "create_matmul_preference");
    cublas(cublasLtMatmulPreferenceSetAttribute(matmul_preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &WORKSPACE_SIZE, sizeof(WORKSPACE_SIZE)), "create_matmul_preference");
    nvtxRangePop();
    return matmul_preference;
}

void free_matmul_preference(cublasLtMatmulPreference_t matmul_preference) {
    nvtxRangePush("free_matmul_preference");
    cublas(cublasLtMatmulPreferenceDestroy(matmul_preference), "free_matmul_preference");
    nvtxRangePop();
}

void *create_workspace(size_t workspace_size) {
    nvtxRangePush("create_workspace");
    void *d_workspace;
    cuda(cudaMalloc(&d_workspace, workspace_size), "create_workspace");
    nvtxRangePop();
    return d_workspace;
}

void free_workspace(void *d_workspace) {
    nvtxRangePush("free_workspace");
    cuda(cudaFree(d_workspace), "free_workspace");
    nvtxRangePop();
}

cudaStream_t create_stream() {
    nvtxRangePush("create_stream");
    cudaStream_t stream;
    cuda(cudaStreamCreate(&stream), "create_stream");
    nvtxRangePop();
    return stream;
}

void free_stream(cudaStream_t stream) {
    nvtxRangePush("free_stream");
    cuda(cudaStreamDestroy(stream), "free_stream");
    nvtxRangePop();
}

cublasLtMatrixLayout_t create_matrix_layout(matrix_t *matrix) {
    nvtxRangePush("create_matrix_layout");
    cublasLtMatrixLayout_t matrix_layout;
    auto height = matrix->height;
    auto width = matrix->width;
    cublas(cublasLtMatrixLayoutCreate(&matrix_layout, DATA_TYPE, height, width, height), "create_matrix_layout");
    nvtxRangePop();
    return matrix_layout;
}

void free_matrix_layout(cublasLtMatrixLayout_t matrix_layout) {
    nvtxRangePush("free_matrix_layout");
    cublas(cublasLtMatrixLayoutDestroy(matrix_layout), "free_matrix_layout");
    nvtxRangePop();
}

cublasLtMatrixTransformDesc_t create_matrix_transform_desc() {
    nvtxRangePush("create_matrix_transform_desc");
    cublasLtMatrixTransformDesc_t matrix_transform_desc;
    cublas(cublasLtMatrixTransformDescCreate(&matrix_transform_desc, DATA_TYPE), "create_matrix_transform_desc");
    nvtxRangePop();
    return matrix_transform_desc;
}

void free_matrix_transform_desc(cublasLtMatrixTransformDesc_t matrix_transform_desc) {
    nvtxRangePush("free_matrix_transform_desc");
    cublas(cublasLtMatrixTransformDescDestroy(matrix_transform_desc), "free_matrix_transform_desc");
    nvtxRangePop();
}

context_t *create_context() {
    nvtxRangePush("create_context");
    auto context = (context_t *) malloc(sizeof(context_t));
    context->handle = create_handle();
    context->matmul_desc = create_matmul_desc();
    context->matmul_preference = create_matmul_preference();
    context->matrix_transform_desc = create_matrix_transform_desc();
    context->d_workspace = create_workspace(WORKSPACE_SIZE);
    context->workspace_size = WORKSPACE_SIZE;
    context->stream = create_stream();
    nvtxRangePop();
    return context;
}

void free_context(context_t *context) {
    nvtxRangePush("free_context");
    free_handle(context->handle);
    free_matmul_desc(context->matmul_desc);
    free_matmul_preference(context->matmul_preference);
    free_matrix_transform_desc(context->matrix_transform_desc);
    free_workspace(context->d_workspace);
    free_stream(context->stream);
    free(context);
    nvtxRangePop();
}


// ========== CUBLASLT AND MISCELLANEOUS ==========

matmul_heuristic_cache_t MATMUL_HEURISTIC_CACHE;
mutex MATMUL_HEURISTIC_MUTEX;

unused_matrices_t UNUSED_MATRICES;
mutex UNUSED_MATRICES_MUTEX;

bool append_unused_matrix(matrix_t *matrix) {
    nvtxRangePush("append_unused_matrix");
    nvtxRangePush("acquiring_unused_matrices_mutex");
    lock_guard lock(UNUSED_MATRICES_MUTEX);
    nvtxRangePop();
    auto unused_matrices = &UNUSED_MATRICES;
    auto key = std::make_tuple(matrix->height, matrix->width);
    if (unused_matrices->count(key) == 0) {
        unused_matrices->insert({key, stack<matrix_t *>()});
    }
    auto entry = &unused_matrices->at(key);
    if (entry->size() < MAX_UNUSED_MATRICES_SAVED) {
        entry->push(matrix);
        nvtxRangePop();
        return true;
    } else {
        nvtxRangePop();
        return false;
    }
}

matrix_t *get_unused_matrix(int height, int width) {
    nvtxRangePush("get_unused_matrix");
    nvtxRangePush("acquiring_unused_matrices_mutex");
    lock_guard lock(UNUSED_MATRICES_MUTEX);
    nvtxRangePop();
    auto unused_matrices = &UNUSED_MATRICES;
    auto key = std::make_tuple(height, width);
    if (unused_matrices->count(key) == 0) {
        unused_matrices->insert({key, stack<matrix_t *>()});
    }
    auto entry = &unused_matrices->at(key);
    if (entry->size() > 0) {
        auto res = entry->top();
        entry->pop();
        nvtxRangePop();
        return res;
    } else {
        nvtxRangePop();
        return nullptr;
    }
}

matrix_t *import_matrix(NeuralNetworkFloat *matrix_array, int height, int width) {
    nvtxRangePush("import_matrix");
    auto matrix = get_matrix(height, width);
    auto array_size = height * width * sizeof(NeuralNetworkFloat);
    cuda(cudaMemcpy(matrix->d_matrix_array, matrix_array, array_size, cudaMemcpyHostToDevice), "import_matrix");
    nvtxRangePop();
    return matrix;
}

NeuralNetworkFloat *export_matrix(matrix_t *matrix, int *height, int *width) {
    nvtxRangePush("export_matrix");
    auto array_size = matrix->height * matrix->width * sizeof(NeuralNetworkFloat);
    auto matrix_array = (NeuralNetworkFloat *) malloc(array_size);
    cuda(cudaMemcpy(matrix_array, matrix->d_matrix_array, array_size, cudaMemcpyDeviceToHost),
         "export_matrix");
    *height = matrix->height;
    *width = matrix->width;
    return matrix_array;
}

matrix_t *create_matrix(NeuralNetworkFloat value, int height, int width) {
    nvtxRangePush("create_matrix");
    auto len = height * width;
    auto matrix_array = (NeuralNetworkFloat *) malloc(len * sizeof(NeuralNetworkFloat));
    for (int i = 0; i < len; i++) {
        matrix_array[i] = value;
    }
    auto matrix = import_matrix(matrix_array, height, width);
    nvtxRangePop();
    return matrix;
}

matrix_t *clone_matrix(matrix_t *matrix) {
    if (matrix == nullptr) {
        return nullptr;
    }
    nvtxRangePush("clone_matrix");
    auto new_matrix = get_matrix(matrix->height, matrix->width);
    auto array_size = matrix->height * matrix->width * sizeof(NeuralNetworkFloat);
    cuda(cudaMemcpy(new_matrix->d_matrix_array, matrix->d_matrix_array, array_size, cudaMemcpyDeviceToDevice),
         "clone_matrix");
    nvtxRangePop();
    return new_matrix;
}

void free_matrix(matrix_t *matrix) {
    if (matrix == nullptr) {
        return;
    }
    nvtxRangePush("free_matrix");
    if (!append_unused_matrix(matrix)) {
        cuda(cudaFree(matrix->d_matrix_array), "free_matrix");
        free(matrix);
    }
    nvtxRangePop();
}

matrix_t *create_empty_matrix(int height, int width) {
    nvtxRangePush("create_empty_matrix");
    auto matrix = (matrix_t *) malloc(sizeof(matrix_t));
    auto array_size = height * width * sizeof(NeuralNetworkFloat);
    cuda(cudaMalloc(&matrix->d_matrix_array, array_size), "create_empty_matrix");
    cuda(cudaMemset(matrix->d_matrix_array, 0, array_size), "create_empty_matrix");
    matrix->height = height;
    matrix->width = width;
    nvtxRangePop();
    return matrix;
}

matrix_t *get_matrix(int height, int width) {
    nvtxRangePush("get_matrix");
    auto res = get_unused_matrix(height, width);
    if (res != nullptr) {
        nvtxRangePop();
        return res;
    } else {
        nvtxRangePop();
        return create_empty_matrix(height, width);
    }
}

NeuralNetworkFloat get(matrix_t *matrix, int row, int column) {
    nvtxRangePush("get");
    NeuralNetworkFloat res = 0.;
    auto index = row + column * matrix->height;
    cuda(cudaMemcpy(&res, &matrix->d_matrix_array[index], sizeof(NeuralNetworkFloat), cudaMemcpyDeviceToHost), "get");
    nvtxRangePop();
    return res;
}

NeuralNetworkFloat *get_result(matrix_t *matrix, int width) {
    nvtxRangePush("get_result");
    assert(matrix->height == 1);
    assert(matrix->width == width);
    auto res = (NeuralNetworkFloat *) malloc(matrix->width * sizeof(NeuralNetworkFloat));
    cuda(cudaMemcpy(res, matrix->d_matrix_array, matrix->width * sizeof(NeuralNetworkFloat), cudaMemcpyDeviceToHost),
         "get_result");
    nvtxRangePop();
    return res;
}

cublasLtMatrixLayout_t create_matrix_layout_from_dimension(int height, int width) {
    nvtxRangePush("create_matrix_layout_from_dimension");
    cublasLtMatrixLayout_t matrix_layout;
    cublas(cublasLtMatrixLayoutCreate(&matrix_layout, DATA_TYPE, height, width, height),
           "create_matrix_layout_from_dimension");
    nvtxRangePop();
    return matrix_layout;
}

cublasLtMatmulHeuristicResult_t compute_matmul_heuristic(context_t *context, cublasLtMatrixLayout_t a_layout,
                                                         cublasLtMatrixLayout_t b_layout,
                                                         cublasLtMatrixLayout_t c_layout) {
    nvtxRangePush("compute_matmul_heuristic");
    auto d_layout = c_layout;
    cublasLtMatmulHeuristicResult_t matmul_heuristic;
    int algo_count;
    cublas(cublasLtMatmulAlgoGetHeuristic(
               context->handle,
               context->matmul_desc,
               a_layout,
               b_layout,
               c_layout,
               d_layout,
               context->matmul_preference,
               1,
               &matmul_heuristic,
               &algo_count
           ), "compute_matmul_heuristic");
    assert(algo_count > 0);
    nvtxRangePop();
    return matmul_heuristic;
}

cublasLtMatmulAlgo_t get_heuristic(context_t *context, int height_lhs, int width_lhs, bool trans_lhs, int height_rhs,
                                   int width_rhs, bool trans_rhs, bool has_bias) {
    nvtxRangePush("get_heuristic");
    nvtxRangePush("acquiring_matmul_heuristic_mutex");
    lock_guard lock(MATMUL_HEURISTIC_MUTEX);
    nvtxRangePop();
    auto heuristic_cache = &MATMUL_HEURISTIC_CACHE;
    auto key = std::make_tuple(height_lhs, width_lhs, trans_lhs, height_rhs, width_rhs, trans_rhs, has_bias);
    if (heuristic_cache->count(key) > 0) {
        auto res = heuristic_cache->at(key);
        nvtxRangePop();
        return res;
    } else {
        auto layout_lhs = create_matrix_layout_from_dimension(height_lhs, width_lhs);
        auto layout_rhs = create_matrix_layout_from_dimension(height_rhs, width_rhs);
        auto layout_res = create_matrix_layout_from_dimension(trans_lhs ? width_lhs : height_lhs,
                                                              trans_rhs ? height_rhs : width_rhs);
        auto res = compute_matmul_heuristic(context, layout_lhs, layout_rhs, layout_res);
        heuristic_cache->insert({key, res.algo});
        nvtxRangePop();
        return res.algo;
    }
}

void set_matmul_desc_epilogue_bias(context_t *context, matrix_t *bias) {
    nvtxRangePush("set_matmul_desc_epilogue_bias");
    auto matmul_desc = context->matmul_desc;
    auto value = CUBLASLT_EPILOGUE_BIAS;
    cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &value, sizeof(value)),
           "set_matmul_desc_epilogue_bias");
    cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias->d_matrix_array,
                                          sizeof(bias->d_matrix_array)), "set_matmul_desc_epilogue_bias");
    nvtxRangePop();
}

void remove_matmul_desc_epilogue_bias(context_t *context) {
    nvtxRangePush("remove_matmul_desc_epilogue_bias");
    auto matmul_desc = context->matmul_desc;
    auto value = CUBLASLT_EPILOGUE_DEFAULT;
    cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &value, sizeof(value)),
           "remove_matmul_desc_epilogue_bias");
    auto value2 = nullptr;
    cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &value2, sizeof(value2)),
           "remove_matmul_desc_epilogue_bias");
    nvtxRangePop();
}

void set_matmul_desc_transpose(context_t *context, bool transpose_lhs, bool transpose_rhs) {
    nvtxRangePush("set_matmul_desc_transpose");
    auto matmul_desc = context->matmul_desc;
    auto value = CUBLAS_OP_T;
    if (transpose_lhs) {
        cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &value, sizeof(value)),
               "set_matmul_desc_transpose");
    }
    if (transpose_rhs) {
        cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &value, sizeof(value)),
               "set_matmul_desc_transpose");
    }
    nvtxRangePop();
}

void remove_matmul_desc_transpose(context_t *context) {
    nvtxRangePush("remove_matmul_desc_transpose");
    auto matmul_desc = context->matmul_desc;
    auto value = CUBLAS_OP_N;
    cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &value, sizeof(value)),
           "remove_matmul_desc_transpose");
    cublas(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &value, sizeof(value)),
           "remove_matmul_desc_transpose");
    nvtxRangePop();
}


// ========== KERNELS ==========

__global__ void apply_sigmoid_kernel(NeuralNetworkFloat *d_matrix, int len) {
    for (int i = get_thread_id(); i < len; i += get_nb_threads()) {
        auto x = d_matrix[i];
        d_matrix[i] = 1. / (1. + exp(-x));
    }
}

__global__ void apply_lhs_rhs_x_times_one_minus_x_then_times_kernel(NeuralNetworkFloat *d_lhs,
                                                                    NeuralNetworkFloat *d_rhs, int len) {
    for (int i = get_thread_id(); i < len; i += get_nb_threads()) {
        auto x = d_rhs[i];
        d_lhs[i] = d_lhs[i] * x * (1. - x);
    }
}

void apply_sigmoid(context_t *context, matrix_t *matrix) {
    nvtxRangePush("apply_sigmoid");
    apply_sigmoid_kernel<<<NB_BLOCS, NB_THREADS_PER_BLOCK, 0, context->stream>>>(
        matrix->d_matrix_array, matrix->height * matrix->width);
    nvtxRangePop();
}

void apply_lhs_rhs_x_times_one_minus_x_then_times(context_t *context, matrix_t *lhs, matrix_t *rhs) {
    nvtxRangePush("apply_lhs_rhs_x_times_one_minus_x_then_times");
    apply_lhs_rhs_x_times_one_minus_x_then_times_kernel<<<NB_BLOCS, NB_THREADS_PER_BLOCK, 0, context->stream>>>(
        lhs->d_matrix_array, rhs->d_matrix_array, lhs->height * lhs->width);
    nvtxRangePop();
}


// ========== MAIN FUNCTIONS ==========

matrix_t *get_layer_output(context_t *context, matrix_t *input, matrix_t *weight, matrix_t *bias) {
    // sigmoid(weight.dot(input) + bias)
    nvtxRangePush("get_layer_output");

    assert(weight->width == input->height);
    auto output = get_matrix(weight->height, input->width);

    auto lhs_layout = create_matrix_layout(weight);
    auto rhs_layout = create_matrix_layout(input);
    auto res_layout = create_matrix_layout(output);

    set_matmul_desc_epilogue_bias(context, bias);

    auto heuristic_algo = get_heuristic(context, weight->height, weight->width, false, input->height, input->width,
                                        false, true);

    nvtxRangePush("get_layer_output matmul");
    cublas(cublasLtMatmul(context->handle, context->matmul_desc, &ONE, weight->d_matrix_array, lhs_layout,
                          input->d_matrix_array, rhs_layout, &ZERO, output->d_matrix_array, res_layout,
                          output->d_matrix_array, res_layout, &heuristic_algo, context->d_workspace,
                          context->workspace_size, context->stream), "get_layer_output");
    nvtxRangePop();

    remove_matmul_desc_epilogue_bias(context);

    free_matrix_layout(lhs_layout);
    free_matrix_layout(rhs_layout);
    free_matrix_layout(res_layout);

    apply_sigmoid(context, output);

    nvtxRangePop();
    return output;
}

matrix_t *dot(context_t *context, matrix_t *lhs, bool transpose_lhs, matrix_t *rhs, bool transpose_rhs) {
    nvtxRangePush("dot");
    auto m = transpose_lhs ? lhs->width : lhs->height;
    auto k = transpose_lhs ? lhs->height : lhs->width;
    assert(k == (transpose_rhs ? rhs->width : rhs->height));
    auto n = transpose_rhs ? rhs->height : rhs->width;
    auto output = get_matrix(m, n);
    auto lhs_layout = create_matrix_layout(lhs);
    auto rhs_layout = create_matrix_layout(rhs);
    auto res_layout = create_matrix_layout(output);

    set_matmul_desc_transpose(context, transpose_lhs, transpose_rhs);

    auto heuristic_algo = get_heuristic(context, lhs->height, lhs->width, transpose_lhs, rhs->height, rhs->width,
                                        transpose_rhs, true);

    nvtxRangePush("dot matmul");
    cublas(cublasLtMatmul(context->handle, context->matmul_desc, &ONE, lhs->d_matrix_array, lhs_layout,
                          rhs->d_matrix_array, rhs_layout, &ZERO, output->d_matrix_array, res_layout,
                          output->d_matrix_array, res_layout, &heuristic_algo, context->d_workspace,
                          context->workspace_size, context->stream), "dot");
    nvtxRangePop();

    remove_matmul_desc_transpose(context);

    free_matrix_layout(lhs_layout);
    free_matrix_layout(rhs_layout);
    free_matrix_layout(res_layout);

    nvtxRangePop();
    return output;
}

void apply_subtract_with_coef(context_t *context, matrix_t *lhs, matrix_t *rhs, NeuralNetworkFloat coef) {
    nvtxRangePush("apply_subtract_with_coef");
    auto lhs_layout = create_matrix_layout(lhs);
    auto rhs_layout = create_matrix_layout(rhs);
    auto output = lhs;
    auto output_layout = lhs_layout;
    auto true_coef = -coef;
    cublas(cublasLtMatrixTransform(context->handle, context->matrix_transform_desc, &ONE, lhs->d_matrix_array,
                                   lhs_layout, &true_coef, rhs->d_matrix_array, rhs_layout, output->d_matrix_array,
                                   output_layout, context->stream), "apply_subtract_with_coef");
    free_matrix_layout(lhs_layout);
    free_matrix_layout(rhs_layout);
    nvtxRangePop();
}
