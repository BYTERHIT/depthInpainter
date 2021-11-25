//
// Created by bytelai on 2021/11/3.
//

#ifndef AUTUFOCUS_RUBUSTINPAITER_H
#define AUTUFOCUS_RUBUSTINPAITER_H

#endif //AUTUFOCUS_RUBUSTINPAITER_H
#include <opencv2/opencv.hpp>
#include "tgvAlgrothm.h"
enum METHOD_INPAINT{
    ALG1 = 0,
    ALG2 = 1,
    ALG_PRECONDITION = 2,
    ALG_TGVL2 = 3,
    ALG_TGVCOLORIZE = 4
};


cv::Mat Inpaint(cv::Mat segImg,cv::Mat depInit, cv::Mat dep, cv::Mat rgbImg, cv::Mat confMap, TGV_PARAM param, METHOD_INPAINT method);
