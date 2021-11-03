//
// Created by laiwenjie on 2021/10/26.
//

#ifndef DEPTHINPAINTER_TGVOPERATOR_H
#define DEPTHINPAINTER_TGVOPERATOR_H
#include <opencv2/opencv.hpp>

typedef struct {
    int idx;//addresss offset
    //grad normerlized grad [dy*dy,-dx*dy,-dy*dx,dx*dx]
    double tGradProjMtx[2][2];//切线方向的投影矩阵
} EDGE_GRAD;
//cv::Mat tgv_alg1(std::vector<EDGE_GRAD> edgeGrad, cv::Mat depth);
cv::Mat tgv_alg2(std::vector<EDGE_GRAD> edgeGrad,cv::Mat depth);

//deprecate
cv::Mat tgv_alg3(std::vector<EDGE_GRAD> edgeGrad,cv::Mat depth);
cv::Mat tgv_alg1(std::vector<EDGE_GRAD> edgeGrad, cv::Mat depth,double lambda_tv, int n_it, double delta, double L);
cv::Mat tgv_algPrecondition(std::vector<EDGE_GRAD> edgeGrad, cv::Mat depth, double lambda_tv, int n_it);
class tgvOperator {

};


#endif //DEPTHINPAINTER_TGVOPERATOR_H
