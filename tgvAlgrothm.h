//
// Created by bytelai on 2021/11/3.
//

#ifndef AUTUFOCUS_TGVALGROTHM_H
#define AUTUFOCUS_TGVALGROTHM_H
#include "tgvOperator.h"
cv::Mat tgv_alg2(std::vector<EDGE_GRAD> edgeGrad,cv::Mat depth, int iterTimes, double lambda_tv, double L);
cv::Mat tgv_alg1(std::vector<EDGE_GRAD> edgeGrad, cv::Mat depth,double lambda_tv, int n_it, double delta, double L);
cv::Mat tgv_algPrecondition(std::vector<EDGE_GRAD> edgeGrad, cv::Mat depth, double lambda_tv, int n_it);
cv::Mat tgv_algTGVL2(cv::Mat spImg, cv::Mat grayImg, cv::Mat depth, double lambda_tv = 0.03, int n_it = 1000);

//deprecate
cv::Mat tgv_alg3(std::vector<EDGE_GRAD> edgeGrad,cv::Mat depth);



#endif //AUTUFOCUS_TGVALGROTHM_H
