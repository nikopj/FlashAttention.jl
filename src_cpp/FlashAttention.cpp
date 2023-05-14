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

void OneDNaive(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long wsize = 0, double lambda = 1.0) {

    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    
    if(wsize == 0)
    {   
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
    else
    {
        long T = N/wsize;
        for (long i = 0; i < T; i++)
        {
            Eigen::MatrixXd O_i = O.block(i*wsize,0,wsize,d);
            OneDNaive(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O_i,0,lambda);
            O.block(i*wsize,0,wsize,d) = O_i;
        }
    }
    
}

void OneDFast(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long cache, long wsize = 0, double lambda = 1.0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    if(wsize != 0)
    {
        for (long i = 0; i < N/wsize; i++)
        {
            Eigen::MatrixXd O_i = O.block(i*wsize,0,wsize,d);
            OneDFast(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O_i, cache, 0, lambda);
            O.block(i*wsize,0,wsize,d) = O_i;
        }
        return;
    }

    const int Bc = ceil(1.0*cache / (4 * d));
    const int Br = min(Bc, d);
    const int Tc = ceil(1.0*N / Bc), Tr = ceil(1.0*N / Br);
    // printf("Tc = %d, Tr = %d, Bc = %d, Br = %d\n",Tc,Tr,Bc,Br);
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
            Eigen::MatrixXd S_ij = Q_i*K_j*lambda;
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

void OneDParallelCPU(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, Eigen::MatrixXd& O, long cache, long wsize = 0, double lambda = 1.0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector

    if(wsize != 0)
    {   
        #pragma omp parallel for
        for (long i = 0; i < N/wsize; i++)
        {
            Eigen::MatrixXd O_i = O.block(i*wsize,0,wsize,d);
            OneDParallelCPU(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O_i, cache, 0, lambda);
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

#pragma omp parallel
{
    #pragma omp for
    for (int i = 0; i < Tr; i++)
    {
        long colr = min(Br, N - i*Br);
        Eigen::MatrixXd Q_i = Q.block(i*Br, 0, colr, d);
        Eigen::MatrixXd O_i = O.block(i*Br, 0, colr, d);
        
        for (int j = 0; j < Tc; j++)
        {
            long colc = min(Bc, N - j*Bc);
            Eigen::VectorXd l_i = l.segment(Br*i, colr);
            Eigen::VectorXd m_i = m.segment(Br*i, colr);
            Eigen::MatrixXd K_j = K.block(j*Bc, 0, colc, d).transpose();
            Eigen::MatrixXd V_j = V.block(j*Bc, 0, colc, d);
            Eigen::MatrixXd S_ij = Q_i*K_j*lambda;
            Eigen::VectorXd m_ij = S_ij.rowwise().maxCoeff();
            Eigen::MatrixXd P_ij = (S_ij.colwise() - m_ij).array().exp().matrix();
            Eigen::VectorXd l_ij = P_ij.rowwise().sum().array();
            Eigen::VectorXd m_new = m_i.cwiseMax(m_ij); 
            Eigen::VectorXd l_new = l_i.array()*((m_i - m_new).array().exp()) + l_ij.array()*((m_ij - m_new).array().exp());
            Eigen::MatrixXd pv = P_ij*V_j;
            for(long i = 0; i < colr; i++)
            {
                O_i.row(i) = (l_i(i)*exp(m_i(i) - m_new(i))*O_i.row(i) + exp(m_ij(i) - m_new(i))*pv.row(i))/l_new(i);
            }
            l.segment(Br*i, colr) = l_new;
            m.segment(Br*i, colr) = m_new;
        }
    O.block(i*Br, 0, colr, d) = O_i;
    }
}
}

void OneDNaiveBack(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXd& dO, 
                    Eigen::MatrixXd& dQ, Eigen::MatrixXd& dK, Eigen::MatrixXd& dV, long wsize = 0, double lambda = 1.0) {

    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector

    if(wsize == 0)
    {   
        dV = P.transpose()*dO;
        Eigen::MatrixXd dP = dO*V.transpose();
        Eigen::VectorXd B = (P.cwiseProduct(dP)).rowwise().sum();
        Eigen::MatrixXd C = (dP.colwise() - B);
        Eigen::MatrixXd dS = P.cwiseProduct(C);
        dQ = dS*K*lambda;
        dK = dS.transpose()*Q*lambda;
    }
    else
    {
        long T = N/wsize;
        for (long i = 0; i < T; i++)
        {
            Eigen::MatrixXd dQ_i = dQ.block(i*wsize,0,wsize,d);
            Eigen::MatrixXd dK_i = dK.block(i*wsize,0,wsize,d);
            Eigen::MatrixXd dV_i = dV.block(i*wsize,0,wsize,d);
            OneDNaiveBack(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),dO.block(i*wsize,0,wsize,d),
            P.block(i*wsize,0,wsize,d),dQ_i,dK_i,dV_i,0,lambda);
            dQ.block(i*wsize,0,wsize,d) = dQ_i;
            dK.block(i*wsize,0,wsize,d) = dK_i;
            dV.block(i*wsize,0,wsize,d) = dV_i;
        }
    }
}

void OneDFastBack(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, const Eigen::MatrixXd& O, const Eigen::MatrixXd& dO, 
                    Eigen::MatrixXd& dQ, Eigen::MatrixXd& dK, Eigen::MatrixXd& dV, const Eigen::VectorXd& l, const Eigen::VectorXd& m, 
                    long cache, long wsize = 0, double lambda = 1.0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    if(wsize != 0)
    {
        long T = N/wsize;
        for (long i = 0; i < T; i++)
        {
            Eigen::MatrixXd dQ_i = dQ.block(i*wsize,0,wsize,d);
            Eigen::MatrixXd dK_i = dK.block(i*wsize,0,wsize,d);
            Eigen::MatrixXd dV_i = dV.block(i*wsize,0,wsize,d);
            OneDFastBack(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O.block(i*wsize,0,wsize,d),
            dO.block(i*wsize,0,wsize,d), dQ_i,dK_i,dV_i,l.segment(i*wsize, 0), m.segment(i*wsize, 0),0,lambda);
            dQ.block(i*wsize,0,wsize,d) = dQ_i;
            dK.block(i*wsize,0,wsize,d) = dK_i;
            dV.block(i*wsize,0,wsize,d) = dV_i;
        }
        return;
    }

    const int Bc = ceil(1.0*cache / (4 * d));
    const int Br = min(Bc, d);
    const int Tc = ceil(1.0*N / Bc), Tr = ceil(1.0*N / Br);

    for (int i = 0; i < Tr; i++)
    {
        long colr = min(Br, N - i*Br);
        Eigen::MatrixXd dQ_i = dQ.block(i*Br, 0, colr, d);
        Eigen::MatrixXd dO_i = dO.block(i*Br, 0, colr, d);
        Eigen::MatrixXd O_i = O.block(i*Br, 0, colr, d);
        Eigen::MatrixXd Q_i = Q.block(i*Br, 0, colr, d);
        Eigen::VectorXd l_i = l.segment(Br*i, colr);
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(l_i.size());
        Eigen::VectorXd m_i = m.segment(Br*i, colr);

        for (int j = 0; j < Tc; j++)
        {
            long colc = min(Bc, N - j*Bc);
            Eigen::MatrixXd K_j = K.block(j*Bc, 0, colc, d).transpose();
            Eigen::MatrixXd V_j = V.block(j*Bc, 0, colc, d);
            Eigen::MatrixXd dK_j = dK.block(j*Bc, 0, colc, d);
            Eigen::MatrixXd dV_j = dV.block(j*Bc, 0, colc, d);
            Eigen::MatrixXd S_ij = lambda*Q_i*K_j;
            Eigen::MatrixXd P_ij = (S_ij.colwise() - m_i).array().exp().matrix();
            // for (long k = 0; k < P_ij.rows(); k++){P_ij.row(k) /= l_i(k);}
            P_ij = P_ij.array().colwise()/l_i.array();
            dV_j = dV_j + P_ij.transpose()*dO_i;
            Eigen::MatrixXd dP_ij = dO_i*V_j.transpose();
            Eigen::VectorXd D_i = (dO_i.cwiseProduct(O_i)).rowwise().sum();
            Eigen::MatrixXd dS_ij = P_ij.cwiseProduct(dP_ij.colwise() - D_i);
            dQ_i = dQ_i + lambda*dS_ij*K_j.transpose();
            dK_j = dK_j +  lambda*dS_ij.transpose()*Q_i;
            dV.block(j*Bc, 0, colc, d) = dV_j;
            dK.block(j*Bc, 0, colc, d) = dK_j;
        }
        dQ.block(i*Br, 0, colr, d) = dQ_i;
    }
}
void OneDParallelCPUBack(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, const Eigen::MatrixXd& O, const Eigen::MatrixXd& dO, 
                    Eigen::MatrixXd& dQ, Eigen::MatrixXd& dK, Eigen::MatrixXd& dV, const Eigen::VectorXd& l, const Eigen::VectorXd& m, 
                    long cache, long wsize = 0, double lambda = 1.0)
{
    const int N = Q.rows(); // number of queries
    const int d = Q.cols(); // dimension of each query/key/value vector
    if(wsize != 0)
    {
        long T = N/wsize;
        #pragma omp parallel for
        for (long i = 0; i < T; i++)
        {
            Eigen::MatrixXd dQ_i = dQ.block(i*wsize,0,wsize,d);
            Eigen::MatrixXd dK_i = dK.block(i*wsize,0,wsize,d);
            Eigen::MatrixXd dV_i = dV.block(i*wsize,0,wsize,d);
            OneDFastBack(Q.block(i*wsize,0,wsize,d), K.block(i*wsize,0,wsize,d),V.block(i*wsize,0,wsize,d),O.block(i*wsize,0,wsize,d),
            dO.block(i*wsize,0,wsize,d), dQ_i,dK_i,dV_i,l.segment(i*wsize, 0), m.segment(i*wsize, 0),0,lambda);
            dQ.block(i*wsize,0,wsize,d) = dQ_i;
            dK.block(i*wsize,0,wsize,d) = dK_i;
            dV.block(i*wsize,0,wsize,d) = dV_i;
        }
        return;
    }

    const int Bc = ceil(1.0*cache / (4 * d));
    const int Br = min(Bc, d);
    const int Tc = ceil(1.0*N / Bc), Tr = ceil(1.0*N / Br);

#pragma omp parallel
{
    #pragma omp for
    for (int i = 0; i < Tr; i++)
    {
        long colr = min(Br, N - i*Br);
        Eigen::MatrixXd dQ_i = dQ.block(i*Br, 0, colr, d);
        Eigen::MatrixXd dO_i = dO.block(i*Br, 0, colr, d);
        Eigen::MatrixXd O_i = O.block(i*Br, 0, colr, d);
        Eigen::MatrixXd Q_i = Q.block(i*Br, 0, colr, d);
        Eigen::VectorXd l_i = l.segment(Br*i, colr);
        Eigen::VectorXd m_i = m.segment(Br*i, colr);

        for (int j = 0; j < Tc; j++)
        {
            long colc = min(Bc, N - j*Bc);
            Eigen::MatrixXd K_j = K.block(j*Bc, 0, colc, d).transpose();
            Eigen::MatrixXd V_j = V.block(j*Bc, 0, colc, d);
            Eigen::MatrixXd dK_j = dK.block(j*Bc, 0, colc, d);
            Eigen::MatrixXd dV_j = dV.block(j*Bc, 0, colc, d);

            Eigen::MatrixXd S_ij = lambda*Q_i*K_j;
            Eigen::MatrixXd P_ij = (S_ij.colwise() - m_i).array().exp().matrix();
            // for (long k = 0; k < P_ij.rows(); k++){P_ij.row(k) /= l_i(k);}
            P_ij = P_ij.array().colwise()/l_i.array();
            dV_j = dV_j + P_ij.transpose()*dO_i;
            Eigen::MatrixXd dP_ij = dO_i*V_j.transpose();
            Eigen::VectorXd D_i = (dO_i.cwiseProduct(O_i)).rowwise().sum();
            Eigen::MatrixXd dS_ij = P_ij.cwiseProduct(dP_ij.colwise() - D_i);
            dQ_i = dQ_i + lambda*dS_ij*K_j.transpose();
            dK_j = dK_j +  lambda*dS_ij.transpose()*Q_i;
            dV.block(j*Bc, 0, colc, d) = dV_j;
            dK.block(j*Bc, 0, colc, d) = dK_j;
        }
        dQ.block(i*Br, 0, colr, d) = dQ_i;
    }
}
}

// int main()
// {
//     Eigen::MatrixXd Q(3,2), K(3,2), V(3,2), dO(3,2),dV(3,2),dQ(3,2), dK(3,2);
//     Q<< 1.2,2.3,
//         4.2,1.1,
//         2.2,2.3;
//     K<< 1.4,2.1,
//         4.6,1,
//         4.2,6.3;
//     V<< 8.2,5.3,
//         1.2,0.1,
//         9.2,4.3;
//     dO<< 0.2,0.3,
//          0.2,0.1,
//          0.2,0.3;
//     Eigen::MatrixXd S(3,3);
//     S.setZero();
//     S = Q*(K.transpose());
//     Eigen::VectorXd Max = S.rowwise().maxCoeff();
//     Eigen::MatrixXd P = (S.colwise() - Max).array().exp().matrix();
//     Eigen::VectorXd l = P.rowwise().sum().array();
//     for(long i = 0; i < P.rows(); i++){P.row(i) /= l(i);}
//     Eigen::MatrixXd O = P*V;

//     dV.setZero();dQ.setZero();dK.setZero();
//     OneDNaiveBack(Q,K,V,P,dO,dQ,dK,dV);
//     cout<<dV<<endl;
//     cout<<dQ<<endl;
//     cout<<dK<<endl;
//     // dV.setZero();dQ.setZero();dK.setZero();
//     // OneDParallelCPUBack(Q, K, V, O, dO, dQ, dK,dV, l, Max, 5);
//     dV.setZero();dQ.setZero();dK.setZero();
//     OneDFastBack(Q, K, V, O, dO, dQ, dK,dV, l, Max, 5);
//     cout<<dV<<endl;
//     cout<<dQ<<endl;
//     cout<<dK<<endl;
//     return 0;
// }

int main()
{
    omp_set_num_threads(omp_get_num_procs()); 
    // omp_set_nested(1);
    // int repeat = 100;
    // int Ns[] = {256, 512, 2048, 4096, 4096*4, 4096*8};
    int repeat = 100;
    int Ns[] = {256,512};
    int Ds[] = {32,64};
    int cache = 24000;

    double tt1 = 0.0, tt2 = 0.0;
    printf("N,d,Naive,FA,pFA,NaiveFlop,FAFlop,pFAFlop,Pass\n");
    for(auto d : Ds)
    {
        for(auto N : Ns)
        {
            //Forward Pass Dense
            Eigen::MatrixXd Q(N,d),K(N,d),V(N,d),O(N,d),O1(N,d),dO(N,d),dQ(N,d),dK(N,d),dV(N,d);
            Q.setRandom();
            K.setRandom();
            V.setRandom();
            dO.setRandom();
            O1.setZero();
            O.setZero();
            dK.setZero();
            dQ.setZero();
            dV.setZero();
            tt1 = omp_get_wtime();
            for(int k = 0; k < repeat; k++)
            {
                OneDNaive(Q, K, V, O1);        
            }
            tt2 = omp_get_wtime();
            double naive = tt2-tt1;
            tt1 = omp_get_wtime();
            for(int k = 0; k < repeat; k++)
            {
                OneDFast(Q,K,V,O,cache);        
            }
            tt2 = omp_get_wtime();
            double FA = tt2 - tt1;
            double eFA = (O.array() - O1.array()).abs().sum();
            tt1 = omp_get_wtime();
            for(int k = 0; k < repeat; k++)
            {
                OneDParallelCPU(Q,K,V,O,cache);        
            }
            tt2 = omp_get_wtime();
            double pFA = tt2 - tt1;
            double epFA = (O.array() - O1.array()).abs().sum();

            double flopsN = 4*repeat*N*N*(d + 11.0/4);
            double flopsFA = 4*repeat*N*N*(d + 11.0/4);

            printf("%d,%d,%f,%f,%f,%f,%f,%f,%s\n",
            N, d, naive/repeat,FA/repeat,pFA/repeat,
            1e-9*flopsN/naive,1e-9*flopsFA/FA,1e-9*flopsFA/pFA,"Forward");

            //Backward Pass Dense

            //Input Prep
            Eigen::MatrixXd S = Q*(K.transpose());
            Eigen::VectorXd Max = S.rowwise().maxCoeff();
            Eigen::MatrixXd P = (S.colwise() - Max).array().exp().matrix();
            Eigen::VectorXd l = P.rowwise().sum().array();
            for(long i = 0; i < P.rows(); i++){P.row(i) /= l(i);}
            // Run starts here
            
            tt1 = omp_get_wtime();
            for(int k = 0; k < repeat; k++)
            {
                dV.setZero();dQ.setZero();dK.setZero(); 
                OneDNaiveBack(Q, K, V, P, dO,dQ,K, dV);        
            }
            tt2 = omp_get_wtime();
            naive = tt2-tt1;
            tt1 = omp_get_wtime();
            for(int k = 0; k < repeat; k++)
            {
                dV.setZero();dQ.setZero();dK.setZero();
                OneDFastBack(Q,K,V,O,dO,dQ,dK,dV,l,Max,cache);        
            }
            tt2 = omp_get_wtime();
            FA = tt2 - tt1;
            tt1 = omp_get_wtime();
            for(int k = 0; k < repeat; k++)
            {
                dV.setZero();dQ.setZero();dK.setZero();
                OneDParallelCPUBack(Q,K,V,O,dO,dQ,dK,dV,l,Max,cache);        
            }
            tt2 = omp_get_wtime();
            pFA = tt2 - tt1;
            flopsN = 4*repeat*N*N*(d + 11.0/4);
            flopsFA = 4*repeat*N*N*(d + 11.0/4);

            printf("%d,%d,%f,%f,%f,%f,%f,%f,%s\n",
            N, d, naive/repeat,FA/repeat,pFA/repeat,
            1e-9*flopsN/naive,1e-9*flopsFA/FA,1e-9*flopsFA/pFA,"Backward");
        }
    }
    return 0;
}