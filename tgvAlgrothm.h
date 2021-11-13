//
// Created by bytelai on 2021/11/3.
//

#ifndef AUTUFOCUS_TGVALGROTHM_H
#define AUTUFOCUS_TGVALGROTHM_H
#include "tgvOperator.h"
typedef struct {
    double lambda;
    double alpha_u;
    double alpha_w;
    double gama;
    double beta;
    int iterTimes;
    double tol;
} TGV_PARAM;

cv::Mat tgv_alg2(std::vector<EDGE_GRAD> edgeGrad,cv::Mat depth, int iterTimes, double lambda_tv, double L);
cv::Mat tgv_alg1(std::vector<EDGE_GRAD> edgeGrad, cv::Mat depth,double lambda_tv, int n_it, double delta, double L);
cv::Mat tgv_algPrecondition(cv::Mat spImg, cv::Mat grayImg, cv::Mat depth, cv::Mat depInit, TGV_PARAM param = {0.025,1.2,17,0.85,9.,1000, 0.1});
cv::Mat tgv_algTGVL2(cv::Mat spImg, cv::Mat grayImg, cv::Mat depth, cv::Mat depInit, TGV_PARAM param = {0.025,1.2,17,0.85,9.,1000,0.1});
cv::Mat tgv_colorizeFiedility(cv::Mat spImg, cv::Mat rgb, cv::Mat depth, cv::Mat depthInit, TGV_PARAM param);
cv::Mat tgv_colorizeFPrecontion(cv::Mat spImg, cv::Mat rgb, cv::Mat dep, cv::Mat depthInit, TGV_PARAM param);
cv::Mat tgv_colorizeFTGVL2(cv::Mat spImg, cv::Mat rgb, cv::Mat depth,cv::Mat depthInit, TGV_PARAM param);

//deprecate
//cv::Mat tgv_alg3(std::vector<EDGE_GRAD> edgeGrad,cv::Mat depth);



#endif //AUTUFOCUS_TGVALGROTHM_H
