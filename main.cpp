#include <iostream>
#include "SLIC.h"
#include "mat_vector.h"
#include "RubustInpaiter.h"
#define DEPTH_SCALE_FACTOR 0.00025
using namespace std;
using namespace cv;


int main() {
    string rootDir = "D:/lwj/projects/DFD_QT/autufocus/out/build/x64-Debug/2021_9_18_14_28_8/";
    Mat img = imread(rootDir + "normal.bmp");
    Mat depth = imread(rootDir+"depinpaintpre.png",IMREAD_UNCHANGED);
    Mat segResult = imread(rootDir+"seg.png",IMREAD_UNCHANGED);
    depth.convertTo(depth,CV_64FC1);
    depth *= DEPTH_SCALE_FACTOR;
    resize(img,img,Size(648,486));
    Mat grayImg;

    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    imwrite("imgResize.bmp", grayImg);

//    SLIC* spMaker = new SLIC(img, depth, 256);
//    spMaker->GreedAggregateSPWithDelpth();
    Mat depInpaint = Inpaint(segResult,depth,img,0.0625,10000,ALG_PRECONDITION);
    depInpaint = depInpaint/DEPTH_SCALE_FACTOR;
    depInpaint.convertTo(depInpaint, CV_16UC1);

    imwrite(rootDir + "tgvInpaintResult.png",depInpaint);
    return 0;
}