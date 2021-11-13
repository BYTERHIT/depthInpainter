//
// Created by bytelai on 2021/11/8.
//

#ifndef DEPTHINPAINTER_SPARSEOPTIMIZER_H
#define DEPTHINPAINTER_SPARSEOPTIMIZER_H
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include "mat_vector.h"
//fidelity J = sum(||U(r)-sum(w(r,s)U(s))||),s belong to r's neighborhood, all w compose A(sparse Matrix,mn*mn)
class SparseOptimizer {
private:
    double _depScale = 1.;
    Eigen::SparseMatrix<double> _DEN;//(I+lambda*T*A'*A)
    Eigen::SparseMatrix<double> _A;//(I+lambda*T*A'*A)
    Eigen::VectorXd _b;
    Eigen::VectorXd _B;//T*A'* b
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> _solver;
public:
    SparseOptimizer(cv::Mat rgb, cv::Mat depth, cv::Mat to, double lambda = 1., double alpha = 1., int winRad = 1);
    SparseOptimizer(cv::Mat rgb, cv::Mat depth, double to, double lambda = 1., double alpha = 1., int winRad = 1);
    cv::Mat GetNewDepth(cv::Mat u_);
    double SparseOptimizer::GetFidelity(cv::Mat u, double lambda);
    mat_vector UpdatePDSteps(cv::Mat to,double alpha = 1.);//利用A更新pd算法的T和yita矩阵，也即步长值
};

cv::Mat MatMulSp(Eigen::SparseMatrix<double> sp, cv::Mat u);

#endif //DEPTHINPAINTER_SPARSEOPTIMIZER_H
