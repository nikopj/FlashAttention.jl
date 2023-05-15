#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#define EIGEN_USE_BLAS
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
using namespace std;
#define BLOCK_SIZE 32

void OneDNaive(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, double lambda = 1.0) {

    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    
    Eigen::MatrixXd S(N,N); // attention scores
    S.setZero();
    // Compute attention scores
    S = lambda*Q*(K.transpose());
    //Softmax scores
    Eigen::VectorXd Max = S.rowwise().maxCoeff();
    Eigen::MatrixXd P = (S.colwise() - Max).array().exp().matrix();
    Eigen::VectorXd l = P.rowwise().sum().array();

    for(long i = 0; i < P.rows(); i++){P.row(i) /= l(i);}

    // Output calculation
    O = P*V;
    
    
}

void OneDNaiveCuda(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long wsize = 0, double lambda = 1.0) {

    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    
    // Initialize CUDA resources
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    
    // Allocate CUDA device memory
    double* S_dev;
    cudaMalloc((void**)&S_dev, N*N*sizeof(double));
    double* P_dev;
    cudaMalloc((void**)&P_dev, N*N*sizeof(double));
    double* Max_dev;
    cudaMalloc((void**)&Max_dev, N*sizeof(double));
    double* l_dev;
    cudaMalloc((void**)&l_dev, N*sizeof(double));
    double* Q_dev;
    double* K_dev;
    double* V_dev;
    double* O_dev;
    cudaMalloc((void**)&Q_dev, d*N*sizeof(double));
    cudaMalloc((void**)&K_dev, d*N*sizeof(double));
    cudaMalloc((void**)&V_dev, d*N*sizeof(double));
    cudaMalloc((void**)&O_dev, d*N*sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(Q_dev, Q.data(), Q.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_dev, K.data(), K.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_dev, V.data(), V.size()*sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute attention scores on GPU
    const double alpha = lambda;
    const double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, d, &alpha, Q_dev, N, K_dev, N, &beta, S_dev, N);
    
    // Compute softmax scores on GPU
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, N, 1);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, N, 1);
    const float alpha_softmax = 1.0;
    const float beta_softmax = 0.0;
    cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha_softmax, x_desc, S_dev, &beta_softmax, y_desc, P_dev);
    
    // Compute output on GPU
    const double alpha_gemm = 1.0;
    const double beta_gemm = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, d, N, &alpha_gemm, P_dev, N, V_dev, d, &beta_gemm, O_dev, N);
    
    // Copy data from device to host
    cudaMemcpy(O.data(), O_dev, O.size()*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free CUDA resources
    cudaFree(S_dev);
    cudaFree(P_dev);
    cudaFree(Max_dev);
    cudaFree(l_dev);
    cublasDestroy(handle);
    cudnnDestroy(cudnn_handle);
}

__global__ void compute_scores_kernel(const float* Q_dev, const float* K_dev, float* S_dev, const int N, const int d, const double lambda, const int Bc, const int Br, const int Tc, const int Tr)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int colc = min(Bc, N - bx*Bc);
    const int colr = min(Br, N - by*Br);
    const int i = by*Br + ty;
    const int j = bx*Bc + tx;
    if (i < N && j < N)
    {
        extern __shared__ float Q_shared[];
        float* K_shared = &Q_shared[Br*d];
        float* S_shared = &K_shared[Bc*d];
        float* m_shared = &S_shared[Br*Bc];
        float* l_shared = &m_shared[Br];
        float* P_shared = &l_shared[Br];
        float* V_shared = &P_shared[Br*Bc];
        float* O_shared = &V_shared[Bc*d];
        float* l_new_shared = &O_shared[Br*d];
        float* m_new_shared = &l_new_shared[Br];
        const int tx_shared = tx % Br;
        const int ty_shared = ty % Bc;
        const int bx_shared = tx / Br;
        const int by_shared = ty / Bc;
        const int i_shared = by_shared*Br + tx_shared;
        const int j_shared = bx_shared*Bc + ty_shared;
        if (tx_shared < colr && ty_shared < colc)
        {
            Q_shared[i_shared*d + tx_shared] = Q_dev[i*d + tx_shared];
            K_shared[ty_shared*d + j_shared] = K_dev[ty_shared*d + j_shared];
            V_shared[ty_shared*d + tx_shared] = V_dev[ty_shared*d + tx_shared];
        }
        __syncthreads();
        if (tx < colr && ty < colc)
        {
            float S_ij = 0.0;
            for (int k = 0; k < d; k++)
            {
                S_ij += Q_shared[i_shared*d + k] * K_shared[k*Bc + ty_shared];
            }
            S_shared[tx_shared*Bc + ty_shared] = S_ij * lambda;
        }
        __syncthreads();
        if (tx_shared < colr && ty_shared < colc)
        {
            float m_ij = -INFINITY;
            for (int k = 0; k < Bc; k++)
            {
                m_ij = fmaxf(m_ij, S_shared[tx_shared*Bc + k]);
            }
            m_shared[tx_shared] = m_ij;
            float l_ij = 0.0;
            for (int k = 0; k < Bc; k++)
            {
                P_shared[tx_shared*Bc + k] = expf(S_shared[tx_shared*Bc + k] - m_ij);
                l_ij += P_shared[tx_shared*Bc + k];
            }
            l_shared[tx_shared] = l_ij;
        }
        __syncthreads();
        if (tx_shared < colr && ty_shared < colc)
        {
            float m_new = -INFINITY;
            for (int k = 0; k < Br; k++)
            {
                m_new = fmaxf(m_new, m_shared[k]);
            }
            m_new_shared[tx_shared] = m_new;
            float l_new = 0.0;
            for (int k = 0; k < Br; k++)
            {
                l_new += l_shared[k] * expf(m_shared[k] - m_new);
            }
            l_new_shared[tx_shared] = l_new;
        }
        __syncthreads();
        if (tx_shared < colr && ty_shared < colc)
        {
            float* O_i_shared = &l_new_shared[Br];
            for (int k = 0; k < d; k++)
            {
                O_i_shared[tx_shared*d + k] = 0.0;
                for (int l = 0; l < Bc; l++)
                {
                    O_i_shared[tx_shared*d + k] += P_shared[tx_shared*Bc + l] * V_shared[l*d + k];
                }
                O_i_shared[tx_shared*d + k] *= expf(m_shared[tx_shared] - m_new_shared[tx_shared]) / l_new_shared[tx_shared];
            }
            for (int k = 0; k < d; k++)
            {
                O_dev[i*d + k] += O_i_shared[tx_shared*d + k];
            }
        }
    }
}

void OneDFast_CUDA(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long cache, double lambda = 1.0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    const int Bc = ceil(1.0*cache / (4 * d));
    const int Br = min(Bc, d);
    const int Tc = ceil(1.0*N / Bc), Tr = ceil(1.0*N / Br);
    // printf("Tc = %d, Tr = %d, Bc = %d, Br = %d\n",Tc,Tr,Bc,Br);
    Eigen::VectorXd l(N),m(N);
    m.setConstant(-numeric_limits<double>::infinity());
    l.setConstant(0.0);

    // Convert Eigen matrices to CUDA device pointers
    const float* Q_dev = Q.data();
    const float* K_dev = K.data();
    const float* V_dev = V.data();
    float* O_dev = O.data();
    
    // Initialize CUDA resources
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    
    // Allocate CUDA device memory
    float* S_dev;
    cudaMalloc((void**)&S_dev, N*N*sizeof(float));
    float* l_dev;
    cudaMalloc((void**)&l_dev, N*sizeof(float));
    float* m_dev;
    cudaMalloc((void**)&m_dev, N*sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(Q_dev, Q.data(), Q.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_dev, K.data(), K.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_dev, V.data(), V.size()*sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute attention scores on GPU
    dim3 dimBlock(Br, Bc);
    dim3 dimGrid(Tc, Tr);
    const int shared_mem_size = (Br*Bc + Br + Bc + 2*Br + Bc*d + Br*sizeof(float) + Bc*sizeof(float)) * sizeof(float);
    compute_scores_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(Q_dev, K_dev, S_dev, N, d, lambda, Bc, Br, Tc, Tr);
    
    // Copy data from device to host
    cudaMemcpy(O.data(), O_dev, O.size()*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free CUDA resources
    cudaFree(S_dev);
    cudaFree(l_dev);
    cudaFree(m_dev);
    cublasDestroy(handle);
    cudnnDestroy(cudnn_handle);
}


int main()
{
    return 0;
}