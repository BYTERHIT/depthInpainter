#include <iostream>
#include "SLIC.h"
#include "mat_vector.h"
#include "RubustInpaiter.h"
#include "ImgSeg.h"
#include "colorize.h"
#define DEPTH_SCALE_FACTOR 0.00025
using namespace std;
using namespace cv;


Mat DepthDotModify(string rootDir, Mat depth) {
    ImgSeg segProc;
    segProc.LoadFromCache(rootDir+"seg.png");
    cv::Mat DepthReplicate;
    map<int, double[4]> segStatistic;
    Mat segGray = segProc.getSegMat();
    Mat regionDepthValid, depthMat;
    threshold(depth, depthMat, 1e2, 1, THRESH_TOZERO_INV);
    depthMat.convertTo(regionDepthValid, CV_8UC1);
    tSEG_INS_MAP segMsks = segProc.getMsks();
    Mat instance, avgIns, stdIns;
    tsInstance segIns;

    for (auto iter = segMsks.begin(); iter!= segMsks.end();iter++) {
        instance = iter->second.msk;
        meanStdDev(depth, avgIns, stdIns, instance);
        double std = max(stdIns.at<double>(0, 0), 0.07);
        segStatistic[iter->first][0] = avgIns.at<double>(0, 0);
        segStatistic[iter->first][1] = stdIns.at<double>(0, 0);
        segStatistic[iter->first][2] = avgIns.at<double>(0, 0) + stdIns.at<double>(0, 0);
        segStatistic[iter->first][3] = avgIns.at<double>(0, 0) - stdIns.at<double>(0, 0);
    }

    for (int i = 0; i < depthMat.rows; i++) {
        for(int j=0; j< depthMat.cols; j++)
        {
            cv::Point coor = Point(j,i);
            float centerVal = depthMat.at<double>(coor);
            int segIdx = segGray.at<uchar>(coor);
            if (centerVal > segStatistic[segIdx][2] || centerVal < segStatistic[segIdx][3] || segIdx == 0) {
                depthMat.at<double>(coor) = 0;
                continue;
            }
        }
    }
    return depthMat;
}
int main() {
    string rootDir = "D:/lwj/projects/DFD_QT/autufocus/cmake-build-release/2021_9_18_16_46_38/";
    Mat img = imread(rootDir + "normal.bmp");
    Mat depth = imread(rootDir+"depthInpaint.png",IMREAD_UNCHANGED);
    Mat depthPredictDfd = imread(rootDir + "depthPredict13.png", IMREAD_UNCHANGED);
    Mat alpha = imread(rootDir + "alpha13.png", IMREAD_UNCHANGED);
    Mat confPredictDfd = imread(rootDir + "confMap13.png", IMREAD_UNCHANGED);
    Mat segResult = imread(rootDir+"seg.png",IMREAD_UNCHANGED);
    Mat depthColorize = imread(rootDir+"colorized.png",IMREAD_UNCHANGED);
    depth.convertTo(depth,CV_64FC1);
    depthColorize.convertTo(depthColorize,CV_64FC1);
    depthPredictDfd.convertTo(depthPredictDfd,CV_64FC1);
    confPredictDfd.convertTo(confPredictDfd,CV_64FC1);
    alpha.convertTo(alpha, CV_64FC1);
    depth *= DEPTH_SCALE_FACTOR;
    depthColorize *= DEPTH_SCALE_FACTOR;
    depthPredictDfd *= DEPTH_SCALE_FACTOR;
    confPredictDfd *= DEPTH_SCALE_FACTOR;
    alpha *= DEPTH_SCALE_FACTOR;
    resize(img,img,Size(648,486));
    Mat grayImg;

    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    imwrite("imgResize.bmp", grayImg);

//    SLIC* spMaker = new SLIC(img, depth, 256);
//    spMaker->GreedAggregateSPWithDelpth();
//    Mat spMat = spMaker->GetSpMap();
//    spMat.convertTo(spMat,CV_8UC1);
    TGV_PARAM param;
    param.iterTimes = 5000;
    param.alpha_w = 5;
    param.alpha_u = 1.2;
    param.lambda = 1e-2;
    param.tol = -1;
    param.gama = 0.85;
    param.beta = 9;
  //  Mat depInpaintColorized = fill_depth_colorization(img,Mat(),depth,1., 1);
    clock_t start, end;
    double duration1,duration2;
    double maxDep,minDep;
    minMaxLoc(depthColorize,&minDep,&maxDep);
    imshow("colorized img", depthColorize/maxDep);
//    minMaxLoc(depInpaintColorized,&minDep,&maxDep);
 //   imshow("colorizedInpaint img", depInpaintColorized/maxDep);
 //   Mat depColoriedInC = depInpaintColorized/maxDep;
//    depColoriedInC.convertTo(depColoriedInC, CV_8UC1,255.);
 //   imwrite(rootDir + "stgvInpaintResult.png",depColoriedInC);
    start = clock();
    Mat depthPredictDfdFilterd = DepthDotModify(rootDir,depthPredictDfd);
//    Mat depInpaint1 = Inpaint(segResult,depthPredictDfdFilterd,depthPredictDfdFilterd,img, confPredictDfd, param,ALG_TGVL2);
    Mat depInpaint1 = Inpaint(segResult,depth,depth,img, {}, param,ALG_TGVL2);
    end = clock();
    duration1 = (double)(end - start) / CLOCKS_PER_SEC;
    start = clock();
    Mat depInpaint2 = Inpaint(segResult,depth,depth,img,{},param,ALG_TGVCOLORIZE);
    end = clock();
    duration2 = (double)(end - start) / CLOCKS_PER_SEC;
    cout<<"duration1 " <<duration1<<endl;
    cout<<"duration2 " <<duration2<<endl;
//    Mat depInpaint = Inpaint(segResult,depth,depth,img,param,ALG_TGVCOLORIZE);
    depInpaint1 = depInpaint1/DEPTH_SCALE_FACTOR;
    depInpaint1.convertTo(depInpaint1, CV_16UC1);

    imwrite(rootDir + "stgvInpaintResult.png",depInpaint1);
    return 0;
}