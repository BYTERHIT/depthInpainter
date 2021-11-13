//
// Created by bytelai on 2021/11/7.
// 对visiontoolbox中的对于anat levin‘s 算法改进版本的c++实现
//

#ifndef DEPTHINPAINTER_COLORIZE_H
#define DEPTHINPAINTER_COLORIZE_H
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>
typedef struct {
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
} SYS_SPMTX;
SYS_SPMTX GetSysSpMtx(cv::Mat rgb, cv::Mat seg, cv::Mat depth, double alpha, int winRad);
cv::Mat fill_depth_colorization(cv::Mat rgb, cv::Mat seg, cv::Mat depth, double alpha = 1.0, int winRad = 1);


#endif //DEPTHINPAINTER_COLORIZE_H
