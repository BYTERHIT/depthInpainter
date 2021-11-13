//
// Created by bytelai on 2021/11/8.
//

#include "SparseOptimizer.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include "utility.h"
#include "colorize.h"

using namespace cv;
using namespace std;
using namespace Eigen;
SparseMatrix<double> GetSparseMtxDiag(Mat to)
{
    int width = to.cols, height = to.rows, pixNum = width*height;
    SparseMatrix<double> T(pixNum,pixNum);
    vector<Triplet<double>> noZeros;
    double *ptr = (double*)to.data;
    for(int i = 0; i < pixNum;i++)
    {
       noZeros.emplace_back(i,i,*ptr);
       ptr++;
    }
    T.setFromTriplets(noZeros.begin(),noZeros.end());
    return T;
}
VectorXd Mat2Vec(Mat a)
{
    int numPix = a.cols * a.rows;
    VectorXd y(numPix);
    double *ptr = (double *)a.data;
    for (int i=0;i<numPix;i++)
    {
        y[i] = *ptr;
        ptr++;
    }
    return y;
}

Mat Vec2Mat(VectorXd v,int rows, int cols, int type)
{
    Mat tmp(rows,cols,type,v.data());
    Mat ret = tmp.clone();
    return ret;
}

Mat MatMulSp(SparseMatrix<double> sp, Mat u)
{
    VectorXd u_ = Mat2Vec(u);
    VectorXd tmp = sp*u_;
    Mat ret = Vec2Mat(tmp,u.rows,u.cols,u.type());
    return ret;
}

SparseOptimizer::SparseOptimizer(cv::Mat rgb, cv::Mat depth, cv::Mat to, double lambda,double alpha, int winRad){
    int width  = rgb.cols;
    int height = rgb.rows;
    int pixNum = rgb.rows * rgb.cols;
    double maxDep,minDep;
    minMaxLoc(depth,&minDep,&maxDep);
    SYS_SPMTX sysSpmtx = GetSysSpMtx(rgb,Mat(),depth,alpha,winRad);
    SparseMatrix<double> A = sysSpmtx.A;
    VectorXd b = sysSpmtx.b;
    SparseMatrix<double> T;
    Mat to_u;
    SparseMatrix<double> I = GetSparseMtxDiag(Mat::ones(height,width,CV_64FC1));
    if(to.empty())
        T = I;
    else
        T = GetSparseMtxDiag(to);
    SparseMatrix<double> A_t = A.transpose();
    
    _DEN =I + lambda*T*A_t*A;
    _B = T * A_t * b;
    _solver.compute(_DEN);
    if(_solver.info() != Eigen::Success)
    {
        cout<<"eigen solver failed" <<endl;
    }
}

SparseOptimizer::SparseOptimizer(cv::Mat rgb, cv::Mat depth, double to, double lambda,double alpha, int winRad){
    int width  = rgb.cols;
    int height = rgb.rows;
    int pixNum = rgb.rows * rgb.cols;
    double maxDep,minDep;
    minMaxLoc(depth,&minDep,&maxDep);
    _depScale = 1./ maxDep;
    SYS_SPMTX sysSpmtx = GetSysSpMtx(rgb,Mat(),depth,alpha,winRad);
    SparseMatrix<double> A = sysSpmtx.A;
    VectorXd b = sysSpmtx.b;
    SparseMatrix<double> T;
    Mat to_u;
    SparseMatrix<double> I = GetSparseMtxDiag(Mat::ones(height,width,CV_64FC1));
    SparseMatrix<double> A_t = A.transpose();
    _A = A;
    _b = b;

    _DEN =I + lambda*to*A_t*A;
    _B = lambda * to * A_t * b;
    _solver.compute(_DEN);
    if(_solver.info() != Eigen::Success)
    {
        cout<<"eigen solver failed" <<endl;
    }
}

Mat SparseOptimizer::GetNewDepth(cv::Mat u_) {
    VectorXd y = Mat2Vec(u_);
    y = y + _B;
    VectorXd nwb;
    nwb = _solver.solve(y);
    Mat tmp(u_.rows,u_.cols,u_.type(),nwb.data());
    Mat ret = tmp.clone();
    return ret;
}

double SparseOptimizer::GetFidelity(cv::Mat u, double lambda) {

    VectorXd y = Mat2Vec(u);
    VectorXd tmp = _A*y - _b;
    Mat d(u.rows,u.cols,u.type(),tmp.data());
    Mat fidelityMat = d.mul(d);
    double energe = 0.5*lambda*sum(fidelityMat)[0];
    return energe;
}

mat_vector SparseOptimizer::UpdatePDSteps(Mat to,double alpha) {
    Mat to_ = to.clone();
    Mat sigma = Mat::zeros(to.rows,to.cols,to.type());

    for (int k=0; k<_A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(_A, k); it; ++it) {
            double Kij = abs(it.value()); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            double *toPtr = (double *)to_.data;
            *(toPtr + j) += pow(Kij,2-alpha);
            double *sigmaPtr = (double *)sigma.data;
            *(sigmaPtr + i) += pow(Kij,alpha);
        }
    }
    mat_vector ret;
    ret.addItem(to_);
    ret.addItem(sigma);
    return ret;
}