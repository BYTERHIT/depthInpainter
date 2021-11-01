#include <iostream>
#include "SLIC.h"
#include "mat_vector.h"
#include "tgvOperator.h"
#define DEPTH_SCALE_FACTOR 0.00025
using namespace std;
using namespace cv;
//前向差分
vector<EDGE_GRAD> GetSpEdge(Mat spMap, Mat grayImg)
{
    vector<EDGE_GRAD> edgeGradVec;
    Mat img = grayImg.clone();
    double step4nbr[4][2] = { {-1,0},{0,1},{1,0},{0,-1} };
    for (int i = 0; i < spMap.rows - 1; i++) {
        for (int j = 0; j < spMap.cols -1; j++) {
            bool isEdge = false;
            uint16_t yOffset = spMap.at<uint16_t>(i + 1,j) - spMap.at<uint16_t>(i,j);
            uint16_t xOffset = spMap.at<uint16_t>(i, j + 1) - spMap.at<uint16_t>(i,j);
            double yGrad = grayImg.at<uchar>(i + 1,j) - grayImg.at<uchar>(i,j);
            double xGrad = grayImg.at<uchar>(i + 1,j) - grayImg.at<uchar>(i,j);
            if(yOffset !=0 || xOffset != 0)
            {
                double norm2 = sqrt(pow(xGrad,2) + pow(yGrad,2)) + DBL_EPSILON;
                xGrad /= norm2;
                yGrad /= norm2;
                EDGE_GRAD edgeGrad;
                edgeGrad.idx = i * spMap.cols + j;
                //grad normerlized grad [dy*dy,-dx*dy,-dy*dx,dx*dx]
                edgeGrad.tGradProjMtx[0][0] = yGrad * yGrad;
                edgeGrad.tGradProjMtx[0][1] = -xGrad * yGrad;
                edgeGrad.tGradProjMtx[1][0] = -yGrad * xGrad;
                edgeGrad.tGradProjMtx[1][1] = xGrad * xGrad;
                edgeGradVec.push_back(edgeGrad);
                Point p(j, i);
                circle(img, p, 0, Scalar(0, 0, 0), -1);
            }
        }
    }
    imwrite("graySuperPix.jpg",img);
    return edgeGradVec;
}


int main() {
    string rootDir = "D:/lwj/projects/DFD_QT/autufocus/out/build/x64-Debug/2021_10_18_10_56_14/";
    Mat img = imread(rootDir + "normal.bmp");
    Mat depth = imread(rootDir+"depinpaintpre.png",IMREAD_UNCHANGED);
    depth.convertTo(depth,CV_64FC1);
    depth *= DEPTH_SCALE_FACTOR;
    resize(img,img,Size(648,486));
    Mat grayImg;

    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    SLIC* spMaker = new SLIC(img, depth, 256);
    spMaker->GreedAggregateSPWithDelpth();
    vector<EDGE_GRAD> edgeGradV =  GetSpEdge(spMaker->GetSpMap(), grayImg);
//    Mat img = imread("D:/lwj/projects/depthInpainter/data/image.png",IMREAD_UNCHANGED);
//    img.convertTo(img,CV_64FC1);
//    img = img / 255. - 0.5;
//    Mat depInpaint = tgv_alg1({},img,0.03,1000,0.05,12);
//    Mat depInpaint = tgv_alg1({},depth,0.03,1000,0.05,12);
    Mat depInpaint = tgv_alg2({},depth);
    imwrite("result.png",depInpaint);
    return 0;
}