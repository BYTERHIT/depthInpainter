#include <iostream>
#include "SLIC.h"
#include "mat_vector.h"
#include "RubustInpaiter.h"
#include "colorize.h"
#define DEPTH_SCALE_FACTOR 0.00025
using namespace std;
using namespace cv;


int main() {
    string rootDir = "D:/lwj/projects/DFD_QT/autufocus/cmake-build-release/2021_9_18_16_46_38/";
    Mat img = imread(rootDir + "normal.bmp");
    Mat depth = imread(rootDir+"depinpaintpre.png",IMREAD_UNCHANGED);
    Mat segResult = imread(rootDir+"seg.png",IMREAD_UNCHANGED);
    Mat depthColorize = imread(rootDir+"colorized.png",IMREAD_UNCHANGED);
    depth.convertTo(depth,CV_64FC1);
    depthColorize.convertTo(depthColorize,CV_64FC1);
    depth *= DEPTH_SCALE_FACTOR;
    depthColorize *= DEPTH_SCALE_FACTOR;
    resize(img,img,Size(648,486));
    Mat grayImg;

    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    imwrite("imgResize.bmp", grayImg);

    SLIC* spMaker = new SLIC(img, depth, 256);
    spMaker->GreedAggregateSPWithDelpth();
    Mat spMat = spMaker->GetSpMap();
    spMat.convertTo(spMat,CV_8UC1);
    TGV_PARAM param;
    param.iterTimes = 14740;
    param.alpha_w = 5;
    param.alpha_u = 1.2;
    param.lambda = 1e-4;
    param.tol = -1;
    param.gama = 0.85;
    param.beta = 9;
    Mat depInpaintColorized = fill_depth_colorization(img,Mat(),depth,1., 1);
    clock_t start, end;
    double duration1,duration2;
    double maxDep,minDep;
    minMaxLoc(depthColorize,&minDep,&maxDep);
    imshow("colorized img", depthColorize/maxDep);
//    minMaxLoc(depInpaintColorized,&minDep,&maxDep);
    imshow("colorizedInpaint img", depInpaintColorized/maxDep);
    Mat depColoriedInC = depInpaintColorized/maxDep;
    depColoriedInC.convertTo(depColoriedInC, CV_8UC1,255.);
    imwrite(rootDir + "stgvInpaintResult.png",depColoriedInC);
    start = clock();
    Mat depInpaint1 = Inpaint(segResult,depInpaintColorized,depth,img,param,ALG_TGVL2);
    end = clock();
    duration1 = (double)(end - start) / CLOCKS_PER_SEC;
    start = clock();
    Mat depInpaint2 = Inpaint(segResult,depInpaintColorized,depth,img,param,ALG_TGVCOLORIZE);
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