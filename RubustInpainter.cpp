//
// Created by bytelai on 2021/11/3.
//
#include "tgvAlgrothm.h"
#include "RubustInpaiter.h"
using namespace cv;
using namespace std;

#define MIN_NORM_VALUE 1e-8
#define MIN_TENSOR_VAL 1e-8
////前向差分
//mat_vector  GetDGradMtx(Mat grayImg, double gama, double beta)
//{
//    Mat img = grayImg.clone();
//    int width = img.cols, height = img.rows;
//    Mat a = Mat::zeros(height,width,CV_64FC1);
//    Mat b = Mat::zeros(height,width,CV_64FC1);
//    Mat c = Mat::zeros(height,width,CV_64FC1);
//    Mat gradX;
//    Mat gradY;
//    Mat G_x = (Mat_<double>(3,3)<<1,0,-1,2,0,-2,1,0,-1);
//    Mat G_y = G_x.t();
//    filter2D(grayImg,gradX,CV_64FC1,G_x,Point(-1,-1),0,BORDER_REPLICATE);
//    filter2D(grayImg,gradY,CV_64FC1,G_y,Point(-1,-1),0,BORDER_REPLICATE);
//    Mat gradNormL2 = gradX.mul(gradX) + gradY.mul(gradY);
//    sqrt(gradNormL2,gradNormL2);
//
//    divide(gradX,gradNormL2,gradX);
//    divide(gradY,gradNormL2,gradY);
//
//    Mat tmp;
//    pow(gradNormL2,gama,tmp);
//    tmp = -beta * tmp;
//    Mat factor;
//    exp(tmp,factor);
//
//    double * normPtr = (double *) gradNormL2.data;
//    double * dxPtr = (double *) gradX.data;
//    double * dyPtr = (double *) gradY.data;
//    double * ePtr = (double *) factor.data;
//    double *aPtr = (double *)a.data;
//    double *bPtr = (double *)b.data;
//    double *cPtr = (double *)c.data;
//
//    for (int i = 0; i < height * width; i++) {
//        if(*normPtr < MIN_NORM_VALUE)
//        {
//            *dxPtr = 1;
//            *dyPtr = 0;
//        }
//        if(*ePtr<MIN_TENSOR_VAL)
//        {
//            *ePtr = MIN_TENSOR_VAL;
//        }
//
//        double dxdx = (*dxPtr)*(*dxPtr);
//        double dydx = (*dxPtr)*(*dyPtr);
//        double dydy = (*dyPtr)*(*dyPtr);
//        double e = *ePtr;
//        //[a,c;c,b]
//        *aPtr = e * dxdx +  dydy;
//        *cPtr = (e-1)*dydx;
//        *bPtr = e*dydy + dxdx;
//        aPtr++;bPtr++;cPtr++;
//        dxPtr++;dyPtr++;normPtr++;ePtr++;
//    }
//    mat_vector ret;
//    ret.addItem(a);ret.addItem(b);ret.addItem(c);
//    return ret;
//}

//前向差分
vector<EDGE_GRAD> GetSpEdge(Mat spMap, Mat grayImg)
{
    vector<EDGE_GRAD> edgeGradVec;
    Mat img = grayImg.clone();
    Mat G_x = (Mat_<double>(3,3)<<1,0,-1,2,0,-2,1,0,-1);
    Mat G_y = G_x.t();
    Mat gradXImg, gradYImg, gradXSp, gradYSp;
    filter2D(img,gradXImg,CV_64FC1,G_x,Point(-1,-1),0,BORDER_REPLICATE);
    filter2D(img,gradYImg,CV_64FC1,G_y,Point(-1,-1),0,BORDER_REPLICATE);
    Mat gradImgNormL2 = gradXImg.mul(gradXImg) + gradYImg.mul(gradYImg);
    sqrt(gradImgNormL2,gradImgNormL2);
    divide(gradXImg,gradImgNormL2,gradXImg);
    divide(gradYImg,gradImgNormL2,gradYImg);
    filter2D(spMap,gradXSp,CV_64FC1,G_x,Point(-1,-1),0,BORDER_REPLICATE);
    filter2D(spMap,gradYSp,CV_64FC1,G_y,Point(-1,-1),0,BORDER_REPLICATE);
    Mat gradSpNormL2 = gradXSp.mul(gradXSp) + gradYSp.mul(gradYSp);
    sqrt(gradSpNormL2,gradSpNormL2);
    for (int i = 0; i < spMap.rows; i++) {
        for (int j = 0; j < spMap.cols; j++) {
            if(gradSpNormL2.at<double>(i,j) > MIN_NORM_VALUE )
            {
                EDGE_GRAD edgeGrad;
                if(gradImgNormL2.at<double>(i,j) < MIN_NORM_VALUE)
                {
                    gradXImg.at<double>(i,j) = 1;
                    gradYImg.at<double>(i,j) = 0;
                }
                edgeGrad.idx = i * spMap.cols + j;
                double xGrad = gradXImg.at<double>(i,j);
                double yGrad = gradYImg.at<double>(i,j);
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

//Mat Inpaint(Mat rgbImg,Mat dep, double lambda, int iterTimes, METHOD_INPAINT method)
//{
//    Mat grayImg;
//    cvtColor(rgbImg, grayImg, COLOR_BGR2GRAY);
//    int type = dep.type();
//    Mat depIn;
//    if (type != CV_64FC1)
//        dep.convertTo(depIn, CV_64FC1);
//    else
//        depIn = dep.clone();
//    vector<EDGE_GRAD> edgeGradV =  GetDGradMtx(grayImg,5.,1.);
//    Mat depInpaint;
//    switch (method) {
//        case ALG1:
//            depInpaint = tgv_alg1(edgeGradV, depIn, lambda, iterTimes, 0.05, 24);
//            break;
//        case ALG2:
//            depInpaint = tgv_alg2(edgeGradV, depIn, iterTimes, lambda, 24);
//            break;
//        case ALG_PRECONDITION:
//            depInpaint = tgv_algPrecondition(edgeGradV, depIn, lambda, iterTimes);
//            break;
//        default:
//            break;
//    }
//    depInpaint.convertTo(depInpaint, type);
//    return depInpaint;
//
//}
//thx to Robust_Recovery_of_Heavily_Degraded_Depth_Measurements
Mat Inpaint(Mat segImg,Mat dep,Mat rgbImg, double lambda, int iterTimes, METHOD_INPAINT method)
{
    Mat grayImg;
    cvtColor(rgbImg, grayImg, COLOR_BGR2GRAY);
    int type = dep.type();
    Mat depIn;
    if (type != CV_64FC1)
        dep.convertTo(depIn, CV_64FC1);
    else
        depIn = dep.clone();
    vector<EDGE_GRAD> edgeGradV =  GetSpEdge(segImg, grayImg);
    Mat depInpaint;
    switch (method) {
        case ALG1:
            depInpaint = tgv_alg1(edgeGradV, depIn, lambda, iterTimes, 0.05, 24);
            break;
        case ALG2:
            depInpaint = tgv_alg2(edgeGradV, depIn, iterTimes, lambda, 24);
            break;
        case ALG_PRECONDITION:
            depInpaint = tgv_algTGVL2(segImg,grayImg,depIn,0.0625,10000);
//            depInpaint = tgv_algPrecondition(edgeGradV, depIn, lambda, iterTimes);
            break;
        default:
            break;
    }
    depInpaint.convertTo(depInpaint, type);
    return depInpaint;
}