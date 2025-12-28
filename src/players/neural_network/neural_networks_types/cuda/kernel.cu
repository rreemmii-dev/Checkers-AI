#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define NB_THREADS_PER_BLOCK 256
#define NB_BLOCS 256

#define DEBUG 0

void debug(std::string s) {
    if (DEBUG) {
        std::cout << s << std::endl;
    }
}

__device__ int get_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_nb_threads() {
    return gridDim.x * blockDim.x;
}

void cublas(cublasStatus_t status, std::string source) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Error: " << status << std::endl;
        std::cout << "From: " << source << std::endl;
    }
    assert(status == CUBLAS_STATUS_SUCCESS);
}

void cuda(cudaError_t status, std::string source) {
    if (status != cudaSuccess) {
        std::cout << "Error: " << status << std::endl;
        std::cout << "From: " << source << std::endl;
    }
    assert(status == cudaSuccess);
}

typedef struct matrix_s {
    double *d_matrix_array;
    int height;
    int width;
} matrix_t;

// C := alpha * A(**T) * B(**T) + beta * C
void dgemm(
    cublasHandle_t *handle,
    double alpha,
    double beta,
    matrix_t *a,
    bool transpose_a,
    matrix_t *b,
    bool transpose_b,
    matrix_t *c
) {
    cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int m = transpose_a ? a->width : a->height;
    int n = transpose_b ? b->height : b->width;
    int k = transpose_a ? a->height : a->width;
    int k2 = transpose_b ? b->width : b->height;
    assert(k == k2);
    assert(c->height == m);
    assert(c->width == n);
    cublas(cublasDgemm_v2(
               *handle,
               transa,
               transb,
               m,
               n,
               k,
               &alpha,
               a->d_matrix_array,
               a->height,
               b->d_matrix_array,
               b->height,
               &beta,
               c->d_matrix_array,
               c->height
           ), "dgemm");
}

// M := alpha * M
void dscal(
    cublasHandle_t *handle,
    double alpha,
    matrix_t *m
) {
    cublas(cublasDscal_v2(
               *handle,
               m->height * m->width,
               &alpha,
               m->d_matrix_array,
               1
           ), "dscal");
}

// A := A + alpha * B
void daxpy(
    cublasHandle_t *handle,
    double *alpha,
    matrix_t *a,
    matrix_t *b
) {
    int n = a->height * a->width;
    assert(n == b->width * b->height);
    cublas(cublasDaxpy_v2(
               *handle,
               n,
               alpha,
               b->d_matrix_array,
               1,
               a->d_matrix_array,
               1
           ), "daxpy");
}

__global__ void sigmoid_kernel(double *d_out, double *d_in, int len) {
    for (int i = get_id(); i < len; i += get_nb_threads()) {
        double x = d_in[i];
        d_out[i] = 1. / (1. + exp(-x));
    }
}

__global__ void x_times_one_minus_x_kernel(double *d_out, double *d_in, int len) {
    for (int i = get_id(); i < len; i += get_nb_threads()) {
        double x = d_in[i];
        d_out[i] = x * (1 - x);
    }
}

__global__ void times_kernel(double *d_out, double *d_in_a, double *d_in_b, int len) {
    for (int i = get_id(); i < len; i += get_nb_threads()) {
        d_out[i] = d_in_a[i] * d_in_b[i];
    }
}

matrix_t *import_matrix_d_array(
    double *d_matrix_array,
    int height,
    int width
) {
    matrix_t *matrix = (matrix_t *) malloc(sizeof(matrix_t));
    matrix->d_matrix_array = d_matrix_array;
    matrix->height = height;
    matrix->width = width;

    return matrix;
}

extern "C" {
matrix_t *import_matrix_array(
    double *matrix_array,
    int height,
    int width
) {
    double *d_matrix_array;
    size_t size = height * width * sizeof(double);
    cuda(cudaMalloc(&d_matrix_array, size), "import_matrix_array malloc");
    cuda(cudaMemcpy(d_matrix_array, matrix_array, size, cudaMemcpyHostToDevice),
         "import_matrix_array memcpy");

    return import_matrix_d_array(d_matrix_array, height, width);
}

matrix_t *clone_matrix(matrix_t *matrix) {
    if (matrix == NULL) {
        return NULL;
    }
    double *d_matrix_array;
    size_t size = matrix->height * matrix->width * sizeof(double);
    cuda(cudaMalloc(&d_matrix_array, size), "clone_matrix malloc");
    cuda(cudaMemcpy(d_matrix_array, matrix->d_matrix_array, size,
                    cudaMemcpyDeviceToDevice), "clone_matrix memcpy");

    return import_matrix_d_array(d_matrix_array, matrix->height, matrix->width);
}

matrix_t *create_matrix(int height, int width, double default_value) {
    double *matrix_array = (double *) malloc(height * width * sizeof(double));
    for (int i = 0; i < height * width; i++) {
        matrix_array[i] = default_value;
    }
    matrix_t *res = import_matrix_array(matrix_array, height, width);
    free(matrix_array);
    return res;
}

void free_matrix(matrix_t *matrix) {
    if (matrix == NULL) {
        return;
    }
    cuda(cudaFree(matrix->d_matrix_array), "free_matrix");
    free(matrix);
}

double get(matrix_t *m, int i, int j) {
    // Column-major ordering is required by cuBLAS
    double res;
    cuda(cudaMemcpy(&res, &m->d_matrix_array[i + j * m->height], sizeof(double), cudaMemcpyDeviceToHost), "get");
    return res;
}

cublasHandle_t *create_handle() {
    cublasHandle_t *handle = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
    cublas(cublasCreate_v2(handle), "create_handle");
    return handle;
}

void free_handle(cublasHandle_t *handle) {
    cublas(cublasDestroy_v2(*handle), "free_handle");
    free(handle);
}

matrix_t *plus(cublasHandle_t *handle, matrix_t *a, matrix_t *b) {
    double alpha = 1.;
    matrix_t *res = clone_matrix(a);
    daxpy(
        handle,
        &alpha,
        res,
        b
    );
    return res;
}

matrix_t *scale(cublasHandle_t *handle, double alpha, matrix_t *m) {
    matrix_t *res = clone_matrix(m);
    dscal(
        handle,
        alpha,
        res);
    return res;
}

matrix_t *dot(cublasHandle_t *handle, matrix_t *a, bool t_a, matrix_t *b, bool t_b) {
    int height = t_a ? a->width : a->height;
    int width = t_b ? b->height : b->width;
    matrix_t *res = create_matrix(height, width, 0.);
    dgemm(
        handle,
        1.,
        0.,
        a,
        t_a,
        b,
        t_b,
        res);
    return res;
}

matrix_t *subtract_with_coef(cublasHandle_t *handle, matrix_t *a, matrix_t *b, double coef) {
    double alpha = -1. * coef;
    matrix_t *res = clone_matrix(a);
    daxpy(
        handle,
        &alpha,
        a,
        b);
    return res;
}

matrix_t *sigmoid(matrix_t *m) {
    matrix_t *res = clone_matrix(m);
    sigmoid_kernel<<<NB_BLOCS, NB_THREADS_PER_BLOCK>>>(res->d_matrix_array, m->d_matrix_array, m->height * m->width);
    cuda(cudaDeviceSynchronize(), "sigmoid");
    return res;
}

matrix_t *x_times_one_minus_x(matrix_t *m) {
    matrix_t *res = clone_matrix(m);
    x_times_one_minus_x_kernel<<<NB_BLOCS, NB_THREADS_PER_BLOCK>>>(res->d_matrix_array, m->d_matrix_array,
                                                                   m->height * m->width);
    cuda(cudaDeviceSynchronize(), "x_times_one_minus_x");
    return res;
}

matrix_t *times(matrix_t *x, matrix_t *y) {
    matrix_t *res = clone_matrix(x);
    times_kernel<<<NB_BLOCS, NB_THREADS_PER_BLOCK>>>(res->d_matrix_array, x->d_matrix_array, y->d_matrix_array,
                                                     x->height * x->width);
    cuda(cudaDeviceSynchronize(), "times");
    return res;
}
}
