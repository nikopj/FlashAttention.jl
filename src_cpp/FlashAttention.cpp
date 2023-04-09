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
using namespace std;

void OneDNaive(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long wsize = 0) {

    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    

    if(wsize == 0)
    {   
        Eigen::MatrixXd S(N,N); // attention scores
        S.setZero();
        // Compute attention scores
        S = Q*(K.transpose());
        //Softmax scores
        Eigen::VectorXd Max = S.rowwise().maxCoeff();
        Eigen::MatrixXd P = (S.colwise() - Max).array().exp().matrix();
        Eigen::VectorXd l = P.rowwise().sum().array();

        for(long i = 0; i < P.rows(); i++){P.row(i) /= l(i);}

        // Output calculation
        O = P*V;
    }
    else
    {
        long T = N/wsize;
        for (long i = 0; i < T; i++)
        {
            Eigen::MatrixXd O_i = O.block(i*wsize,0,wsize,d);
            OneDNaive(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O_i,0);
            O.block(i*wsize,0,wsize,d) = O_i;
        }
    }
    
}

void OneDFast(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long cache, long wsize = 0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    if(wsize != 0)
    {
        for (long i = 0; i < N/wsize; i++)
        {
            Eigen::MatrixXd O_i = O.block(i*wsize,0,wsize,d);
            OneDFast(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O_i, cache, 0);
            O.block(i*wsize,0,wsize,d) = O_i;
        }
        return;
    }

    const int Bc = ceil(1.0*cache / (4 * d));
    const int Br = min(Bc, d);
    const int Tc = ceil(1.0*N / Bc), Tr = ceil(1.0*N / Br);
    Eigen::VectorXd l(N),m(N);
    m.setConstant(-numeric_limits<double>::infinity());
    l.setConstant(0.0);

    for (int j = 0; j < Tc; j++)
    {
        long colc = min(Bc, N - j*Bc);
        Eigen::MatrixXd K_j = K.block(j*Bc, 0, colc, d).transpose();
        Eigen::MatrixXd V_j = V.block(j*Bc, 0, colc, d);
        for (int i = 0; i < Tr; i++)
        {
            long colr = min(Br, N - i*Br);
            Eigen::MatrixXd Q_i = Q.block(i*Br, 0, colr, d);
            Eigen::VectorXd l_i = l.segment(Br*i, colr);
            Eigen::VectorXd m_i = m.segment(Br*i, colr);
            Eigen::MatrixXd S_ij = Q_i*K_j;
            Eigen::VectorXd m_ij = S_ij.rowwise().maxCoeff();
            Eigen::MatrixXd P_ij = (S_ij.colwise() - m_ij).array().exp().matrix();
            Eigen::VectorXd l_ij = P_ij.rowwise().sum().array();
            Eigen::VectorXd m_new = m_i.cwiseMax(m_ij); 
            Eigen::VectorXd l_new = l_i.array()*((m_i - m_new).array().exp()) + l_ij.array()*((m_ij - m_new).array().exp());
            Eigen::MatrixXd O_i = O.block(i*Br, 0, colr, d);
            Eigen::MatrixXd pv = P_ij*V_j;
            for(long i = 0; i < colr; i++)
            {
                O_i.row(i) = (l_i(i)*exp(m_i(i) - m_new(i))*O_i.row(i) + exp(m_ij(i) - m_new(i))*pv.row(i))/l_new(i);
            }
            O.block(i*Br, 0, colr, d) = O_i;
            l.segment(Br*i, colr) = l_new;
            m.segment(Br*i, colr) = m_new;
        }
    }
}

void OneDParallelCPU(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long cache, long wsize = 0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector

    if(wsize != 0)
    {   
        #pragma omp parallel
{       
        #pragma omp for
        for (long i = 0; i < N/wsize; i++)
        {
            Eigen::MatrixXd O_i = O.block(i*wsize,0,wsize,d);
            OneDParallelCPU(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O_i, cache, 0);
            O.block(i*wsize,0,wsize,d) = O_i;
        }
}
        return;
    }

    const int Bc = ceil(1.0*cache / (4 * d));
    const int Br = min(Bc, d);
    const int Tc = ceil(1.0*N / Bc), Tr = ceil(1.0*N / Br);
    Eigen::VectorXd l(N),m(N);
    m.setConstant(-numeric_limits<double>::infinity());
    l.setConstant(0.0);

#pragma omp parallel
{ 
#pragma omp for
    for (int j = 0; j < Tc; j++)
    {
        long colc = min(Bc, N - j*Bc);
        Eigen::MatrixXd K_j = K.block(j*Bc, 0, colc, d).transpose();
        Eigen::MatrixXd V_j = V.block(j*Bc, 0, colc, d);
        for (int i = 0; i < Tr; i++)
        {
            long colr = min(Br, N - i*Br);
            Eigen::MatrixXd Q_i = Q.block(i*Br, 0, colr, d);
            Eigen::VectorXd l_i = l.segment(Br*i, colr);
            Eigen::VectorXd m_i = m.segment(Br*i, colr);
            Eigen::MatrixXd S_ij = Q_i*K_j;
            Eigen::VectorXd m_ij = S_ij.rowwise().maxCoeff();
            Eigen::MatrixXd P_ij = (S_ij.colwise() - m_ij).array().exp().matrix();
            Eigen::VectorXd l_ij = P_ij.rowwise().sum().array();
            Eigen::VectorXd m_new = m_i.cwiseMax(m_ij); 
            Eigen::VectorXd l_new = l_i.array()*((m_i - m_new).array().exp()) + l_ij.array()*((m_ij - m_new).array().exp());
            Eigen::MatrixXd O_i = O.block(i*Br, 0, colr, d);
            Eigen::MatrixXd pv = P_ij*V_j;
            for(long i = 0; i < colr; i++)
            {
                O_i.row(i) = (l_i(i)*exp(m_i(i) - m_new(i))*O_i.row(i) + exp(m_ij(i) - m_new(i))*pv.row(i))/l_new(i);
            }
            O.block(i*Br, 0, colr, d) = O_i;
            l.segment(Br*i, colr) = l_new;
            m.segment(Br*i, colr) = m_new;
        }
    }

}
}

int main()
{
    // int N = 3, d = 3;
    // Eigen::MatrixXd Q(N,d),K(N,d),V(N,d),O(N,d),O1(N,d);
    // Q<< 2,1,2,
    //     1,2,1,
    //     2,1,2;
    // K<< 4,3,4,
    //     3,4,3,
    //     4,3,4;
    // V<< 6,5,6,
    //     5,6,5,
    //     6,5,6;

    // O1.setZero();
    // O.setZero();
    // OneDNaive(Q,K,V,O1);
    // OneDFast(Q,K,V,O,12);
    
    // cout<<O1<<endl;
    // cout<<O<<endl;

    omp_set_num_threads(omp_get_num_procs());
    int Ns[] = {2048*256*4};
    int Ds[] = {32};
    int caches[] = {2048*16, 2048*256};
    double tt1 = 0.0, tt2 = 0.0, tt3 = 0.0;
    printf("N     d    Fast      parallel   cacheFast  cacheParallel\n");
    for(auto d : Ds)
    {
        for(auto N : Ns)
        {
            Eigen::MatrixXd Q(N,d),K(N,d),V(N,d),O(N,d),O1(N,d);
            Q.setRandom();
            K.setRandom();
            V.setRandom();
            O1.setZero();
            O.setZero();
            // printf("I am done!\n");
            // tt1 = omp_get_wtime();
            // OneDNaive(Q, K, V, O1);
            // tt2 = omp_get_wtime();
            // double naive = tt2-tt1;
            double minimT = numeric_limits<double>::infinity(), minimError = 0.0;
            double minimT2 = numeric_limits<double>::infinity(), minimError2 = 0.0;
            long minimCache = 0, minimCache2 = 0;
            for(auto cache : caches)
            {
                tt1 = omp_get_wtime();
                OneDFast(Q,K,V,O,cache, 64);
                tt2 = omp_get_wtime();
                double error = (O.array() - O1.array()).abs().sum();
                if(minimT > tt2-tt1)
                {
                    minimT = tt2-tt1;
                    minimCache = cache;
                    minimError = error;
                }
                tt1 = omp_get_wtime();
                OneDParallelCPU(Q,K,V,O,cache, 64);
                tt2 = omp_get_wtime();
                error = (O.array() - O1.array()).abs().sum();
                if(minimT2 > tt2-tt1)
                {
                    minimT2 = tt2-tt1;
                    minimCache2 = cache;
                    minimError2 = error;
                }
            }
            // printf("%d  %d  %fs  %fs  %ld  %ld  %f\n", N, d, naive, minimT, minimT2, long(ceil(1.0*4*d*N/minimCache)), minimError);
            printf("%d  %d  %fs  %fs  %ld    %ld\n", N, d, minimT, minimT2, minimCache ,minimCache2);
        }
    }
   return 0;
}