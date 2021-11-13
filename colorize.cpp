//
// Created by bytelai on 2021/11/7.
//

#include "colorize.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include "tgvOperator.h"
#include "utility.h"

using namespace std;
using namespace cv;
using namespace Eigen;

Mat calculateIndxTable(int rows, int cols)
{
    Mat table = Mat::zeros(rows,cols,CV_32SC1);
    int* ptr = (int*)table.data;
    int idx = 0;
    for(int i = 0; i < rows * cols; i++) {
        *ptr = i;
        ptr++;
    }
    return table;
}

SYS_SPMTX GetSysSpMtx(Mat rgb, Mat seg, Mat depth, double alpha, int winRad)
{
    int width = rgb.cols;
    int height = rgb.rows;
    int pixNum = rgb.cols*rgb.rows;
    static Mat indsM = calculateIndxTable(height, width);
    SparseMatrix<double> A(pixNum,pixNum);
    SparseMatrix<double> G(pixNum,pixNum);
    VectorXd b(pixNum);
    vector<Triplet<double>> nonZeros, nonZerosG,nonZerosB;

//    MAX_MIN_NORM normDep = MaxMinNormalizeNoZero(depth);
    double maxDep;// = normDep.max;
    double minDep;// = normDep.min;
//    Mat depN = normDep.norm;
    cv::minMaxLoc(depth,&minDep,&maxDep);
    Mat depN = depth / maxDep;
    int winSize = pow(2*winRad + 1,2);
    mat_vector coeffs(winSize,Mat::zeros(height,width,CV_64FC1));

    Mat gray;
    cvtColor(rgb,gray,COLOR_BGR2GRAY);
    gray.convertTo(gray,CV_64FC1,1./255);
    int absImgNdx = 0;
    for(int j = 0; j < width; j++)
    {
        absImgNdx = j;
        for(int i = 0; i < height; i++)
        {
            Mat avg,stdvar;
            int winXStart = max(0, j - winRad), winXEnd = min(width, j + winRad+1);
            int winYStart = max(0, i - winRad), winYEnd = min(height, i + winRad+1);
            Rect winRect = Rect(Point(winXStart,winYStart),Point(winXEnd,winYEnd));
            int16_t curSegIdx;
            Mat aroundSeg;
            Mat msk;
            if(!seg.empty())
            {
                curSegIdx = seg.at<uchar>(i,j);
                aroundSeg = seg(winRect).clone();
                aroundSeg.convertTo(aroundSeg,CV_16SC1);
                aroundSeg = abs(aroundSeg-curSegIdx);
                threshold(aroundSeg,msk,0,1,THRESH_BINARY_INV);
                msk.convertTo(msk,CV_8UC1);
            }
            meanStdDev(gray(winRect),avg,stdvar,msk);
            double c_var = pow(stdvar.at<double>(0,0),2);
            double csig = c_var * 0.6;

            double curVal = gray.at<double>(i,j);
            Mat aroundVal = gray(winRect).clone() - curVal;
            
            pow(aroundVal,2,aroundVal);
            double mgv = MinNoZero(aroundVal,msk);
            double mgv_ = -mgv / LN1EM2;
            if(csig < mgv_)
                csig = mgv_;
            if(csig < 2e-6)
                csig = 2e-6;
            aroundVal = aroundVal / ( - csig);
            exp(aroundVal, aroundVal);
            if (!msk.empty()) {
                Mat tmp;
                msk.convertTo(tmp, CV_64FC1);
                aroundVal = aroundVal.mul(tmp);
            }
            double sum_;
            sum_ = sum(aroundVal)[0] - 1.;//1 是(i,j)位置的值(exp(0)=1)
            aroundVal = aroundVal / sum_;
            double curDep = depN.at<double>(i, j);
            for(int ii = winYStart; ii < winYEnd; ii++)
            {
                for(int jj = winXStart; jj < winXEnd; jj++)
                {
                    if (ii == i && jj == j)
                    {
                        double diagValue = 1.0;
                        if (curDep > 0)
                            diagValue += alpha;
                        nonZeros.emplace_back(absImgNdx, absImgNdx, diagValue);
                    }
                    else
                    {
                        nonZeros.emplace_back(absImgNdx,indsM.at<int>(ii,jj),-aroundVal.at<double>(ii - winYStart,jj - winXStart));
                        int winIdx = (ii - i + winRad ) * ( 2 * winRad + 1) + (jj - j + winRad);
                        coeffs[winIdx].at<double>(i,j) = aroundVal.at<double>(ii - winYStart,jj - winXStart);
                    }
                }
            }
            b[absImgNdx] = alpha * curDep;
            absImgNdx += width;
        }
    }
    A.setFromTriplets(nonZeros.begin(),nonZeros.end());
    SYS_SPMTX ret;
    ret.A = A;
    ret.b = b;
    return ret;
}

Mat fill_depth_colorization(Mat rgb, Mat seg, Mat depth, double alpha, int winRad)
{
    int height = rgb.rows;
    int width = rgb.cols;
    int pixNum = rgb.cols * rgb.rows;
    double maxDep, minDep;
    minMaxLoc(depth,&minDep,&maxDep);
    SYS_SPMTX sysSpmtx = GetSysSpMtx(rgb,seg,depth,alpha,winRad);
    VectorXd nwb(pixNum);

//最小二乘解超静定方程组
    SparseMatrix<double> A = sysSpmtx.A;
    VectorXd b = sysSpmtx.b;
    SparseMatrix<double> A_transpose = A.transpose();
    SparseMatrix<double> AAt = A_transpose*A;


    Eigen::SimplicialLLT<SparseMatrix<double>> solver;// (AAt);
//    Eigen::SparseQR <Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
//    Eigen::SparseLU<SparseMatrix<double>> solver;
//    Eigen::BiCGSTAB<SparseMatrix<double>> solver;
//    Eigen::LeastSquaresConjugateGradient<SparseMatrix<double>> solver;


    Eigen::VectorXd b_ = A_transpose*b;
    solver.compute(AAt);
    if(solver.info() != Eigen::Success)
    {
        cout<<"eigen solver failed" <<endl;
        return Mat();
    }
    nwb = solver.solve(b_);
    Mat tmp(height,width,CV_64FC1, nwb.data());
    Mat ret = tmp.clone();
    ret = ret * maxDep;
    return ret;
}