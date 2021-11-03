//
// Created by laiwenjie on 2021/10/26.
//

#include "tgvOperator.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "mat_vector.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
//#define d_u_w

using namespace cv;
using namespace std;
using namespace Eigen;


Point2d projL1(Point2d p2d, double alpha = 1.0)
{
    double norm = abs(p2d.x)+abs(p2d.y);
    norm = std::max(1.0,norm/alpha);
    return p2d/norm;
}
Point2d projL2(Point2d p2d, double alpha = 1.0)
{
    double norm = sqrt(p2d.x*p2d.x+p2d.y*p2d.y);
    norm = std::max(1.0,norm/alpha);
    return p2d/norm;
}
//前向差分,输入是一个通道的深度数据
mat_vector  derivativeForward(Mat input)
{
    mat_vector difVec;
    Rect xRoi1 = Rect(0,0,input.cols-1,input.rows);
    Rect xRoi2 = Rect(1,0,input.cols-1,input.rows);
    Rect yRoi1 = Rect(0,0,input.cols,input.rows-1);
    Rect yRoi2 = Rect(0,1,input.cols,input.rows-1);


    Mat xDif = Mat::zeros(input.rows,input.cols,input.type());
    Mat yDif = Mat::zeros(input.rows,input.cols,input.type());
    xDif(xRoi1) = input(xRoi2) - input(xRoi1);
    yDif(yRoi1) = input(yRoi2) - input(yRoi1);
    difVec.addItem(xDif);
    difVec.addItem(yDif);
    return difVec;
}
//利用反向差分的负（和前向差分共轭）散度和差分是负共轭关系
Mat divergence(mat_vector grad )
{
    Mat xGrad = grad[0];
    Mat yGrad = grad[1];
    int height = xGrad.rows;
    int width = xGrad.cols;
    int type = xGrad.type();

    Rect fistCol = Rect(0,0,1,height);
    Rect back2Col = Rect(width-2,0,1,height);
    Rect xRoi1 = Rect(0,0,width-2,height);
    Rect xRoi2 = Rect(1,0,width-2,height);

    Rect fistRow = Rect(0,0,width,1);
    Rect back2Row = Rect(0,height-2,width,1);
    Rect yRoi1 = Rect(0,0,width,height-2);
    Rect yRoi2 = Rect(0,1,width,height-2);
    Mat div = Mat::zeros(height,width,type);

    div(fistCol)+=xGrad(fistCol);
    div(xRoi2)+=xGrad(xRoi2)-xGrad(xRoi1);
    div(back2Col)-=xGrad(back2Col);
    div(fistRow)+=yGrad(fistRow);
    div(yRoi2)+=yGrad(yRoi2)-yGrad(yRoi1);
    div(back2Row)-=yGrad(back2Row);
    return div;
}

mat_vector symmetrizedSecondDerivative(mat_vector grad)
{
    mat_vector sym2ndDif;
    Mat xGrad = grad[0];
    Mat yGrad = grad[1];
    mat_vector difX = derivativeForward(xGrad);
    mat_vector difY = derivativeForward(yGrad);
    sym2ndDif.addItem(difX[0]);//xx
    sym2ndDif.addItem((difX[1]+difY[0])/2);//(xy+yx)/2  xy
    sym2ndDif.addItem((difX[1]+difY[0])/2);//(xy+yx)/2  yx
    sym2ndDif.addItem(difY[1]);//yy
    return sym2ndDif;
}
//TODO div2 symmetrizedSecondDerivative 的共轭算子
mat_vector second_order_divergence(mat_vector second_order_derivative){
    mat_vector vec;
    Mat xxGrad = second_order_derivative[0];
    Mat xyGrad = second_order_derivative[1];
    Mat yxGrad = second_order_derivative[2];
    Mat yyGrad = second_order_derivative[3];
    Mat xy_yx = (xyGrad + yxGrad)*0.5;
    int height = xxGrad.rows;
    int width = xxGrad.cols;
    int type = xxGrad.type();

    Rect fistCol = Rect(0,0,1,height);
    Rect back2Col = Rect(width-2,0,1,height);
    Rect xRoi1 = Rect(0,0,width-2,height);
    Rect xRoi2 = Rect(1,0,width-2,height);

    Rect fistRow = Rect(0,0,width,1);
    Rect back2Row = Rect(0,height-2,width,1);
    Rect yRoi1 = Rect(0,0,width,height-2);
    Rect yRoi2 = Rect(0,1,width,height-2);

    Mat xDiv = Mat::zeros(height,width,type);
    xDiv(fistCol) += xxGrad(fistCol);
    xDiv(xRoi2) += xxGrad(xRoi2)-xxGrad(xRoi1);
    xDiv(back2Col) -= xxGrad(back2Col);
    xDiv(fistRow) += xy_yx(fistRow);
    xDiv(yRoi2) += xy_yx(yRoi2)-xy_yx(yRoi1);
    xDiv(back2Row) -= xy_yx(back2Row);

    Mat yDiv = Mat::zeros(height,width,type);
    yDiv(fistCol)+=xy_yx(fistCol);
    yDiv(xRoi2)+=xy_yx(xRoi2)- xy_yx(xRoi1);
    yDiv(back2Col)-=xy_yx(back2Col);
    yDiv(fistRow)+=yyGrad(fistRow);
    yDiv(yRoi2)+=yyGrad(yRoi2)-yyGrad(yRoi1);
    yDiv(back2Row)-=yyGrad(back2Row);

    vec.addItem(xDiv);
    vec.addItem(yDiv);
    return vec;
}
//D*dU
//grad normerlized grad [dy*dy,-dx*dy,-dy*dx,dx*dx]
//edgePos pos = y*width + x;
mat_vector D_OPERATOR(vector<EDGE_GRAD> edgeGrad, mat_vector du)
{
    if(edgeGrad.empty())
    {
        return du;
    }
    Mat uDx = du[0];
    Mat uDy = du[1];
    mat_vector vec;
    Mat dx = du[0].clone();
    Mat dy = du[1].clone();
    double* uDxPtr = (double*)uDx.data;
    double* uDyPtr = (double*)uDy.data;
    double* xPtr = (double*)dx.data;
    double* yPtr = (double*)dy.data;
    for(auto iter = edgeGrad.begin();iter!=edgeGrad.end();iter++)
    {
        int pos = iter->idx;
        double iDx2AtPos = iter->tGradProjMtx[1][1];
        double iDy2AtPos = iter->tGradProjMtx[0][0];
        double iDxDyAtPos = iter->tGradProjMtx[0][1];
        double uDxAtPos = *(uDxPtr+pos);
        double uDyAtPos = *(uDyPtr+pos);
        *(xPtr + pos) = uDxAtPos*iDy2AtPos + uDyAtPos * iDxDyAtPos;
        *(yPtr + pos) = uDxAtPos*iDxDyAtPos + uDyAtPos * iDx2AtPos;
    }
    vec.addItem(dx);
    vec.addItem(dy);
    return vec;
}
//(I+sigma*dF_star)^-1
// todo 需要确定是用L1还是L2范数，此处先用L1范数
mat_vector F_STAR_OPERATOR(mat_vector pBar, double alpha)
{
    int vecSize = pBar.size();
    mat_vector result;
    int width = pBar[0].cols, height = pBar[0].rows;
    Mat sum = Mat::zeros(height,width,pBar[0].type());
    for(auto iter = pBar.begin(); iter !=pBar.end();iter++)
    {
        sum += iter->mul(*iter);
        //sum += abs(*iter);
    }
    sqrt(sum,sum);
    sum /= alpha;
    double *ptr = (double*)sum.data;
    for(int i = 0 ;i <width*height;i++)
    {
        if(*ptr < 1.)
            *ptr = 1.;
        ptr++;
    }

    for(auto iter = pBar.begin(); iter !=pBar.end();iter++)
    {
        Mat item;
        divide(*iter,sum,item);
        result.addItem(item);
    }
    return result;
}

Mat G_OPERATOR(Mat g, Mat uBar, Mat to, double lambda, double thresh)
{
    Mat u = uBar.clone();
    double *uPtr = (double*)u.data;
    double *uBarPtr = (double*)uBar.data;
    double *gPtr = (double*)g.data;
    double *toPtr = (double*)to.data;
    for(int i = 0;i < uBar.rows*uBar.cols;i++)
    {
        //lamba=0,when g==0
        if(*gPtr > thresh)
            *uPtr = (*uBarPtr + (*toPtr) * lambda * (*gPtr))/(1. + (*toPtr)*lambda);
        else
            *uPtr = *uBarPtr;
        uPtr++;
        uBarPtr++;
        gPtr++;
        toPtr++;
    }
    return u;

}
//(I+to*dG)^-1
Mat G_OPERATOR(Mat g, Mat uBar,double to, double lambda)
{
    Mat u = uBar.clone();
    double *uPtr = (double*)u.data;
    double *uBarPtr = (double*)uBar.data;
    double *gPtr = (double*)g.data;
    for(int i = 0;i < uBar.rows*uBar.cols;i++)
    {
        //lamba=0,when g==0
        if(*gPtr != 0)
            *uPtr = (*uBarPtr + to * lambda * (*gPtr))/(1. + to*lambda);
        else
            *uPtr = *uBarPtr;
        uPtr++;
        uBarPtr++;
        gPtr++;
    }
    return u;
}

double GetEnerge(Mat u,Mat g, mat_vector w, vector<EDGE_GRAD> edgeGrad, double lambda = 1., double alpha0 = 1., double alpha1=1.)
{
    double minDep = 0, maxDep = 10;
    minMaxLoc(g,&minDep,&maxDep);
    Mat offset = u - g;
    Mat mask;
    threshold(g,mask,DBL_EPSILON + minDep,1.,THRESH_BINARY);
    Mat fidelityMat = offset.mul(offset).mul(mask);
    mat_vector div = derivativeForward(u);
    mat_vector divD = D_OPERATOR(edgeGrad,div - w);
    mat_vector dif2 = symmetrizedSecondDerivative(w);
    double tv = div.norm1();
    double tgv = alpha0*divD.norm2() + alpha1*dif2.norm2();

    double energe = 0.5*lambda*sum(fidelityMat)[0] + tgv;//tgv;
    return energe;
}
Mat tgv_alg3(vector<EDGE_GRAD> edgeGrad,Mat depth)
{
    double L = 12.0,alpha_u=1.,alpha_w=2.;
    double lambda = 16., gama = lambda,
    delta = 0.05, mu = 2* sqrt(gama*delta)/L,
    tau_n = mu / (2 * gama), sigma_n = mu / (2*delta), theta_n = 1 / (1 + mu);
    int loopTimes = 1000;
    mat_vector w,p,q,wBar;
    Mat u0 = depth.clone();
    Mat u,uBar;
    Mat w1 = Mat::zeros(depth.rows,depth.cols, depth.type());
    Mat w2 = w1.clone();
    u = w1.clone();
    uBar = depth.clone();

    Mat wBar1= w1.clone();
    Mat wBar2 = w1.clone();
    wBar.addItem(wBar1);
    wBar.addItem(wBar2);
    w.addItem(w1);
    w.addItem(w2);
    Mat dx = w1.clone();
    Mat dy = w1.clone();
    p.addItem(dx);
    p.addItem(dy);
    Mat dxx = w1.clone();
    Mat dxy = w1.clone();
    Mat dyx = w1.clone();
    Mat dyy = w1.clone();
    q.addItem(dxx);
    q.addItem(dxy);
    q.addItem(dyx);
    q.addItem(dyy);

    for(int i = 0; i<loopTimes; i++)
    {
        mat_vector du = derivativeForward(uBar) - wBar;
        du = D_OPERATOR(edgeGrad, du);
        p = p+du*sigma_n;
        p= F_STAR_OPERATOR(p,alpha_u);

        q = q + symmetrizedSecondDerivative(wBar)*sigma_n;
        q = F_STAR_OPERATOR(q,alpha_w);

        Mat u_old = u.clone();
        mat_vector dp = D_OPERATOR(edgeGrad,p);
        Mat uDelta = divergence(dp) * tau_n;
        u = u + uDelta;
        u = G_OPERATOR(u0,u,tau_n,lambda);
        uBar = u + (u-u_old)*theta_n;

        mat_vector w_old = w.clone();
        w = second_order_divergence(q) + p;
        w = w + w * tau_n;
        wBar = w + (w-w_old)*theta_n;

//        theta_n = 1 / sqrt( 1 + 2 * gama * tau_n);
//        tau_n = theta_n * tau_n;
//        sigma_n = sigma_n / theta_n;
        double energe = GetEnerge(u,u0,w,edgeGrad,lambda,alpha_u,alpha_w);

        namedWindow("depthInpaint");
        Mat uGray = u * 256.;
        uGray.convertTo(uGray,CV_8UC1);
        imshow("depthInpaint",uGray);
        waitKey(10);
        cout<<"iter:"<<i<<" loss: "<<energe<<endl;
    }
    return u;
}

Point GetCoor(int idx ,int rows, int cols)
{
    int i = idx / cols;
    int j = idx % rows;
    return Point(j,i);
}

/* 对于图像m行n列，前向差分算子[DX;DY]:
 * |-----------------------------m block----------------------------------------------------------|
 *  |--n cols--|
 *  -1  1      |
 *      .  .   |
 *        -1  1|
 *            0|
 * -------------------------
 *             |-1  1      |
 *             |    .  .   |
 *             |      -1  1|
 *             |          0|
 *             -------------
 *                               *
 *                                     *
 *                                           |------------
 *                                           |-1  1      |
 *                                           |    .  .   |
 *                                           |      -1  1|
 *                                           |          0|
 * -------------------------                 |------------
 *  -1         | 1         |
 *      .      |    .      |
 *         .   |       .   |
 *           -1|          1|
 * ------------------------------------------------------
 *                   *           *
 *                         *           *
 * ------------------------------|------------------------
 *                               |-1         | 1         |
 *                               |    .      |    .      |
 *                               |       .   |       .   |
 *                               |         -1|          1|
 *                               -------------------------
 *                                           | 0         |
 *                                           |    .      |
 *                                           |       .   |
 *                                           |          0|
 *  对于D_EDGE，D=[A11,A12;A21,A22]
 * */
mat_vector GetSteps(vector<EDGE_GRAD> edgeGrad, int rows, int cols, double alpha_u, double alpha_w, double alpha = 1.)
{
    int size = rows * cols;
    SparseMatrix<double> firstTwoRowOfBlocks(2*size,3*size);
    SparseMatrix<double> K(6*size,3*size);
    SparseMatrix<double> D_edge(2*size,2*size);
    vector<Triplet<double>> nonZeros;
    //K:
    //{alpha_u*D_edge*[dx,      -I,               0;
    //                 dy,       0,              -I]
    //                 0,    alpha_w*dx,          0;
    //                 0,    alpha_w/2*dy,  alpha_w/2*dx;
    //                 0,    alpha_w/2*dy,  alpha_w/2*dx;
    //                 0,        0,            alpha_w*dy}
    for(int j = 0;j<rows;j++){//dx
        int offset = j*cols;
        for(int i = 0; i < cols -1;i++)
        {
            nonZeros.emplace_back(offset + i, offset + i,-1.);
            nonZeros.emplace_back( offset + i, offset + i + 1,1.);
        }
    }
    for(int j = 0;j<rows -1;j++){//dy
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZeros.emplace_back(offset + size + i, offset + i,-1.);
            nonZeros.emplace_back( offset + size + i, offset + i + cols,1.);
        }
    }
#ifdef d_u_w
    for(int j = 0;j<rows;j++){//-I 0;0 -I;
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZeros.emplace_back(offset + i, offset + size + i,-1.);
            nonZeros.emplace_back( offset + i + size, offset + 2*size + i, -1.);
        }
    }
#endif
    firstTwoRowOfBlocks.setFromTriplets(nonZeros.begin(),nonZeros.end());

    vector<Triplet<double>> nonZerosDedge;
    if (!edgeGrad.empty())
    {
        auto iter = edgeGrad.begin();
        int idx = iter->idx;
        for (int j = 0; j < size; j++) {
            if (j != idx)
            {
                nonZerosDedge.emplace_back(j, j, 1.);//xx
                nonZerosDedge.emplace_back(j + size, j + size, 1.);//yy
            }
            else
            {
                nonZerosDedge.emplace_back(j, j, iter->tGradProjMtx[0][0]);//xx
                nonZerosDedge.emplace_back(j, j + size, iter->tGradProjMtx[0][1]);//xy
                nonZerosDedge.emplace_back(j + size, j, iter->tGradProjMtx[1][0]);//yx
                nonZerosDedge.emplace_back(j + size, j + size, iter->tGradProjMtx[1][1]);//yy
                if (iter < edgeGrad.end() - 1)
                {
                    iter++;
                    idx = iter->idx;
                }
            }
        }
    }
    else
    {
        for (int j = 0; j < size; j++) {
            nonZerosDedge.emplace_back(j, j, 1.);//xx
            nonZerosDedge.emplace_back(j + size, j + size, 1.);//yy
        }
    }
    
    D_edge.setFromTriplets(nonZerosDedge.begin(),nonZerosDedge.end());

    firstTwoRowOfBlocks = alpha_u * D_edge * firstTwoRowOfBlocks;
    vector<Triplet<double>> nonZerosK;
    for (int k=0; k<firstTwoRowOfBlocks.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(firstTwoRowOfBlocks, k); it; ++it) {
            double Kij = it.value(); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            nonZerosK.emplace_back(i,j,Kij);
        }
    }

    //delta x
    for (int j = 0; j < rows; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols - 1; i++) {
            nonZerosK.emplace_back(size*2 + offset + i, size + offset + i, -alpha_w);
            nonZerosK.emplace_back(size*2 + offset + i, size + offset + i + 1, alpha_w);

            nonZerosK.emplace_back(size*3 + offset + i, size*2 + offset + i  , -0.5 * alpha_w);
            nonZerosK.emplace_back(size*3 + offset + i, size*2 + offset + i   + 1, 0.5 * alpha_w);

            nonZerosK.emplace_back(size*4 + offset + i, size*2 + offset + i  , -0.5 * alpha_w);
            nonZerosK.emplace_back(size*4 + offset + i, size*2 + offset + i   + 1, 0.5 * alpha_w);
        }
    }
    //delta y
    for (int j = 0; j < rows - 1; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols; i++) {
            nonZerosK.emplace_back(offset + i + 3*size, size + offset + i, -0.5 * alpha_w);
            nonZerosK.emplace_back(offset + i + 3*size, size + offset + i + cols, 0.5 * alpha_w);

            nonZerosK.emplace_back(offset + i + 4*size, size + offset + i, -0.5 * alpha_w);
            nonZerosK.emplace_back(offset + i + 4*size, size + offset + i + cols, 0.5 * alpha_w);

            nonZerosK.emplace_back(offset + i + 5*size, offset + 2*size + i, -alpha_w);
            nonZerosK.emplace_back(offset + i + 5*size, offset + 2*size + i + cols, alpha_w);
        }
    }
#ifndef d_u_w
    for(int j = 0;j<rows;j++){//-I 0;0 -I;
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZerosK.emplace_back(offset + i, offset + size + i,-alpha_u);
            nonZerosK.emplace_back( offset + i + size, offset + 2*size + i, -alpha_u);
        }
    }
#endif
    K.setFromTriplets(nonZerosK.begin(), nonZerosK.end());
//    cout << K << endl;
    //迭代访问稀疏矩阵
    mat_vector to(3,Mat::zeros(rows,cols,CV_64FC1));
    mat_vector sigma(6,Mat::zeros(rows,cols,CV_64FC1));
    for (int k=0; k<K.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
            double Kij = abs(it.value()); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            int toMatIdx = j / size;
            int toOffset = j % size;
            double *toPtr = (double *)to[toMatIdx].data;
            *(toPtr + toOffset) += pow(Kij,2-alpha);
            int sigmaMatIdx = i / size;
            int sigmaOffset = i % size;
            double *sigmaPtr = (double *)sigma[sigmaMatIdx].data;
            *(sigmaPtr + sigmaOffset) += pow(Kij,alpha);
        }
    }
    mat_vector params;
    Mat ones = Mat::ones(rows, cols,CV_64FC1);
    //Mat xoffset = sigmaX - sigmaX_;
    //Mat yoffset = sigmaY - sigmaY_;
    //Mat tooffset = to - to_;

    double delta = DBL_EPSILON;
    for(int i = 0; i<3;i++)
    {
        Mat toTmp;
        divide(ones,to[i] + delta,toTmp);
        params.addItem(toTmp);
    }
    for(int i = 0; i<6; i++)
    {
        Mat tmp;
        divide(ones,sigma[i] + delta,tmp);
        params.addItem(tmp);
    }
    return params;
}

mat_vector GetUstepsUsingMat(vector<EDGE_GRAD> edgeGrad, int rows, int cols)
{
    //debug code to seee grad
    Mat A11 = Mat::ones(rows, cols, CV_64FC1),
        A12 = Mat::zeros(rows, cols, CV_64FC1),
        A21 = A12.clone(),
        A22 = A11.clone();
    double* a11Ptr = (double*)A11.data;
    double* a12Ptr = (double*)A12.data;
    double* a21Ptr = (double*)A21.data;
    double* a22Ptr = (double*)A22.data;
    for (auto it = edgeGrad.begin(); it != edgeGrad.end(); it++)
    {
        *(a11Ptr + it->idx) = it->tGradProjMtx[0][0];
        *(a12Ptr + it->idx) = it->tGradProjMtx[0][1];
        *(a21Ptr + it->idx) = it->tGradProjMtx[1][0];
        *(a22Ptr + it->idx) = it->tGradProjMtx[1][1];
    }
    Mat A11Dx, A12Dy, A21Dx, A22Dy;
    //对m行n列的图像，其差分算子DX，DY是mn行mn列的稀疏矩阵，只有在以下位置(坐标序号为(row,col))有值
    Mat DXDiag; //对角线元素
    Mat DYDiag;//对角线
    Mat DYDiagURN;//(0,n)->(mn-n,mn)
    Mat DXDiagUR1 = Mat::ones(rows, cols, CV_64FC1);//(0,1)->(mn-1,mn)
    DYDiag = DXDiagUR1.clone();
    DYDiagURN = DYDiag.clone();
    DXDiag = -1 * Mat::ones(rows, cols, CV_64FC1);
    Rect lastCol = Rect(cols - 1, 0, 1, rows);
    Rect lastRow = Rect(0, rows - 1, cols, 1);
    Rect ur1Region = Rect(1, 0, cols - 1, rows);
    Rect ur1RegionAnchor = Rect(0, 0, cols - 1, rows);
    Rect urnRegion = Rect(0, 1, cols, rows - 1);
    Rect urnRegionAnchor = Rect(0, 0, cols, rows - 1);
    DXDiag(lastCol) *= 0;
    DXDiagUR1(lastCol) *= 0;
    DYDiag(lastRow) *= 0;
    DYDiag *= -1;
    DYDiagURN(lastRow) *= 0;
    Mat K11Diag, K11DiagUR1, K11DiagURN, K21Diag, K21DiagURN, K21DiagUR1;
    K11Diag = A11.mul(DXDiag) + A12.mul(DYDiag);
    K11DiagUR1 = A11.mul(DXDiagUR1);
    K11DiagURN = A12.mul(DYDiagURN);
    K21Diag = A21.mul(DXDiag) + A22.mul(DYDiag);
    K21DiagUR1 = A21.mul(DXDiagUR1);
    K21DiagURN = A22.mul(DYDiagURN);

    Mat sigmaX_ = abs(K11Diag) + abs(K11DiagUR1) + abs(K11DiagURN);
    Mat sigmaY_ = abs(K21Diag) + abs(K21DiagUR1) + abs(K21DiagURN);
    Mat to_ = abs(K11Diag) + abs(K21Diag);
    to_(ur1Region) += abs(K11DiagUR1(ur1RegionAnchor)) + abs(K21DiagUR1(ur1RegionAnchor));
    to_(urnRegion) += abs(K11DiagURN(urnRegionAnchor)) + abs(K21DiagURN(urnRegionAnchor));

    mat_vector params;
    Mat ones = Mat::ones(rows, cols, CV_64FC1);
    //Mat xoffset = sigmaX - sigmaX_;
    //Mat yoffset = sigmaY - sigmaY_;
    //Mat tooffset = to - to_;

    divide(ones, to_ + DBL_EPSILON, to_);
    divide(ones, sigmaX_ + DBL_EPSILON, sigmaX_);
    divide(ones, sigmaY_ + DBL_EPSILON, sigmaY_);
    params.addItem(to_);
    params.addItem(sigmaX_);
    params.addItem(sigmaY_);
    return params;
}
/*
 * w=[w1;w2],2mn*1;
 * eps(w) = [dx 0; dy/2 dx/2; dy/2 dx/2; 0 dy]*w
 */
mat_vector GetWSteps(int rows, int cols) {
    int size = rows * cols;
    SparseMatrix<double> EPSILON(4 * size, 2 * size), K(2 * size, size);
    vector<Triplet<double>> nonZeros;
    //delta x
    for (int j = 0; j < rows; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols - 1; i++) {
            nonZeros.emplace_back(offset + i, offset + i, -1.);
            nonZeros.emplace_back(offset + i, offset + i + 1, 1.);

            nonZeros.emplace_back(offset + i + size, offset + i + size, -0.5);
            nonZeros.emplace_back(offset + i + size, offset + i + size + 1, 0.5);

            nonZeros.emplace_back(offset + i + 2*size, offset + i + size, -0.5);
            nonZeros.emplace_back(offset + i + 2*size, offset + i + size + 1, 0.5);
        }
    }
    //delta y
    for (int j = 0; j < rows - 1; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols; i++) {
            nonZeros.emplace_back(offset + i + size, offset + i, -0.5);
            nonZeros.emplace_back(offset + i + size, offset + i + cols, 0.5);

            nonZeros.emplace_back(offset + i + 2*size, offset + i, -0.5);
            nonZeros.emplace_back(offset + i + 2*size, offset + i + cols, 0.5);

            nonZeros.emplace_back(offset + i + 3*size, offset + size + i, -1);
            nonZeros.emplace_back(offset + i + 3*size, offset + size + i + cols, 1);
        }
    }
    EPSILON.setFromTriplets(nonZeros.begin(),nonZeros.end());
//    cout <<"EPSILON:" << EPSILON << endl;
    cout <<"迭代访问稀疏矩阵的元素 "<<endl;
    Mat toX = Mat::zeros(rows,cols,CV_64FC1);
    Mat toY = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaXX = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaXY = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaYX = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaYY = Mat::zeros(rows,cols,CV_64FC1);
    double *toXPtr = (double*)toX.data;
    double *toYPtr = (double*)toY.data;
    double *sigmaXXPtr = (double*)sigmaXX.data;
    double *sigmaXYPtr = (double*)sigmaXY.data;
    double *sigmaYXPtr = (double*)sigmaYX.data;
    double *sigmaYYPtr = (double*)sigmaYY.data;
    double alpha = 1.;
    for (int k=0; k< EPSILON.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(EPSILON, k); it; ++it) {
            double Kij = abs(it.value()); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            if(j>=size)
                *(toYPtr + j - size) += pow(Kij, 2-alpha);
            else
                *(toXPtr + j) += pow(Kij,2-alpha);
            if(i >= 3*size)
                *(sigmaYYPtr + i - 3*size) += pow(Kij,alpha);
            else if(i>=2*size)
                *(sigmaYXPtr + i - 2*size) += pow(Kij,alpha);
            else if(i>=size)
                *(sigmaXYPtr + i - size) += pow(Kij,alpha);
            else
                *(sigmaXXPtr + i) += pow(Kij,alpha);
        }
    }
    mat_vector ret;
    Mat ones = Mat::ones(rows, cols,CV_64FC1);
    divide(ones,toX + DBL_EPSILON,toX);
    divide(ones,toY + DBL_EPSILON,toY);
    divide(ones,sigmaXX + DBL_EPSILON,sigmaXX);
    divide(ones,sigmaXY + DBL_EPSILON,sigmaXY);
    divide(ones,sigmaYX + DBL_EPSILON,sigmaYX);
    divide(ones,sigmaYY + DBL_EPSILON,sigmaYY);
    ret.addItem(toX);
    ret.addItem(toY);
    ret.addItem(sigmaXX);
    ret.addItem(sigmaXY);
    ret.addItem(sigmaYX);
    ret.addItem(sigmaYY);
    return ret;
}
Mat tgv_algPrecondition(vector<EDGE_GRAD> edgeGrad, Mat depth, double lambda_tv = 0.03, int n_it = 1000)
{
    double minDep = 0, maxDep = 10;
    minMaxLoc(depth,&minDep,&maxDep);
    double scaleDep = 1. / (maxDep - minDep);
    depth = depth * scaleDep - minDep * scaleDep - 0.5;

    double theta_n = 1 ;
    double alpha_u = 2., alpha_w = 1.;
    //初始化步长值 通过precontion算法实现
    mat_vector steps = GetSteps(edgeGrad,depth.rows, depth.cols,alpha_u,alpha_w,1.);
    Mat to_u = steps[0];
    mat_vector to_w(2);
    copy(steps.begin()+1,steps.begin()+3,to_w.begin());
    mat_vector sigma_p(2);
    copy(steps.begin() + 3, steps.begin()+5,sigma_p.begin());
    mat_vector sigma_q(4);
    copy(steps.begin()+5,steps.end(),sigma_q.begin());

    double lambda = 1/lambda_tv;
    int loopTimes = n_it;
    Mat zeros = Mat::zeros(depth.rows, depth.cols, depth.type());
    mat_vector w(2,zeros);
    mat_vector wBar(2,zeros);
    mat_vector p(2,zeros);
    mat_vector q(4,zeros);
    Mat u0 = depth.clone();
    Mat u, uBar;
    u =  zeros.clone();
    uBar = u0.clone();
    Mat uGray;

    for (int i = 0; i < loopTimes; i++)
    {
        mat_vector u_bar_grad = derivativeForward(uBar);
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + alpha_u * (D_OPERATOR(edgeGrad,u_bar_grad) - wBar).mul(sigma_p), 1.);
#else
        p = F_STAR_OPERATOR(p + alpha_u * D_OPERATOR(edgeGrad,u_bar_grad - wBar).mul(sigma_p), 1.);
#endif

        mat_vector w_bar_second_derivative = symmetrizedSecondDerivative(wBar);
        q = F_STAR_OPERATOR(q + alpha_w * w_bar_second_derivative.mul(sigma_q), 1.);

        Mat u_old = u.clone();
//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        mat_vector dp = D_OPERATOR(edgeGrad, p);
        Mat p_div = divergence(dp);
        u = G_OPERATOR(u0, u + alpha_u * p_div.mul(to_u), to_u, lambda, -0.5);
        uBar = u + (u - u_old) * theta_n;

        mat_vector w_old = w.clone();
        mat_vector q_second_div = second_order_divergence(q);
        //此处和文献不一样p old?
#ifndef d_u_w
        w = w + (p*alpha_u + alpha_w * q_second_div).mul(to_w);
#else
        w = w + (dp*alpha_u + alpha_w * q_second_div).mul(to_w);
#endif
        wBar = w + (w - w_old) * theta_n;

        double energe = GetEnerge(u, u0, w, edgeGrad, lambda, alpha_u, alpha_w);
        namedWindow("depthInpaint");
        double scale = 1./scaleDep;
        uGray = (u+0.5)*scale + minDep;
//        uGray = (u) * scale + minDep;
        uGray *=100;
        uGray.convertTo(uGray, CV_8UC1);
        imshow("depthInpaint", uGray);
        waitKey(10);
        cout << "iter:" << i << " loss: " << energe << endl;
    }
    return uGray;

}


Mat tgv_alg2(vector<EDGE_GRAD> edgeGrad,Mat depth)
{
    double minDep = 0, maxDep = 10;
    minMaxLoc(depth,&minDep,&maxDep);
    double scaleDep = 1. / (maxDep - minDep);
    depth = depth*scaleDep - minDep * scaleDep;// - 0.5;
    double L = 24.0,alpha_u=2.,alpha_w=1.;
    double to_u=1./L, lambda = 16., gama = lambda, sigma_p = 1./L/L/to_u,theta_u=1/sqrt(1+2*gama*to_u),sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_u;
    int loopTimes = 3000;
    mat_vector w,p,q,wBar;
    Mat u0 = depth.clone();
    Mat u,uBar;
    Mat w1 = Mat::zeros(depth.rows,depth.cols, depth.type());
    Mat w2 = w1.clone();
    u = w1.clone();
    uBar = u0.clone();
    Mat wBar1= w1.clone();
    Mat wBar2 = w1.clone();
    wBar.addItem(wBar1);
    wBar.addItem(wBar2);
    w.addItem(w1);
    w.addItem(w2);
    Mat dx = w1.clone();
    Mat dy = w1.clone();
    p.addItem(dx);
    p.addItem(dy);
    Mat dxx = w1.clone();
    Mat dxy = w1.clone();
    Mat dyx = w1.clone();
    Mat dyy = w1.clone();
    q.addItem(dxx);
    q.addItem(dxy);
    q.addItem(dyx);
    q.addItem(dyy);
    Mat uGray;

    for(int i = 0; i<loopTimes; i++)
    {
        mat_vector u_bar_grad = derivativeForward(uBar);
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + alpha_u * (D_OPERATOR(edgeGrad, u_bar_grad) - wBar)*(sigma_p), 1.);
#else
        p = F_STAR_OPERATOR(p + alpha_u * D_OPERATOR(edgeGrad, u_bar_grad - wBar)*(sigma_p), 1.);
#endif

        mat_vector w_bar_second_derivative = symmetrizedSecondDerivative(wBar);
        q = F_STAR_OPERATOR(q + alpha_w * w_bar_second_derivative*sigma_q,1.);

        Mat u_old = u.clone();
        mat_vector dp = D_OPERATOR(edgeGrad, p);
        Mat p_div = divergence(dp);
        u = G_OPERATOR(u0,u + alpha_u * p_div * to_u,to_u,lambda);
        uBar = u + (u-u_old)*theta_u;

        mat_vector w_old = w.clone();
        mat_vector q_second_div = second_order_divergence(q);

#ifndef d_u_w
        w = w + (p * alpha_u + alpha_w * q_second_div)*(to_w);
#else
        w = w + (dp * alpha_u + alpha_w * q_second_div)*(to_w);
#endif
        wBar = w + (w-w_old)*theta_w;

        theta_u = 1/sqrt(1+2*gama*to_u);
        to_u = theta_u*to_u;
        sigma_p = sigma_p / theta_u;

        theta_w = 1/sqrt(1 + 2 * gama*to_w);
        to_w = theta_w* to_w;
        sigma_q = sigma_q / theta_w;

        double energe = GetEnerge(u,u0,w,edgeGrad,lambda,alpha_u,alpha_w);
        namedWindow("depthInpaint");
        double scale = 1./scaleDep;
        uGray = u*scale + minDep;
        uGray *=50;
        uGray.convertTo(uGray,CV_8UC1);
        imshow("depthInpaint",uGray);
        waitKey(10);
        cout<<"iter:"<<i<<" loss: "<<energe<<endl;
    }
    return uGray;
}

Mat tgv_alg1(vector<EDGE_GRAD> edgeGrad, Mat depth,double lambda_tv, int n_it, double delta, double L)
{
    double minDep = 0, maxDep = 10;
    minMaxLoc(depth,&minDep,&maxDep);
    double scaleDep = 1. / (maxDep - minDep);
    depth = depth*scaleDep - minDep * scaleDep - 0.5;
    double gamma = lambda_tv;
    double mu = 2 * sqrt(gamma * delta) / L;

    double to_u = mu / (2 * gamma);
    double sigma_n = mu / (2 * delta);

    double theta_n = 1 / (1 + mu);

    double alpha_u = 2., alpha_w = 1.;
    double lambda = 1/lambda_tv, gama = lambda;
    double sigma_p = sigma_n;//1. / L / L / to_u, theta_u = 1;// / sqrt(1 + 2 * gama * to_u)
    double sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_n;
    int loopTimes = n_it;
    mat_vector w, p, q, wBar;
    Mat u0 = depth.clone();
    Mat u, uBar;
    Mat w1 = Mat::zeros(depth.rows, depth.cols, depth.type());
    Mat w2 = w1.clone();
    u = w1.clone();
    uBar = u0.clone();
    Mat wBar1 = w1.clone();
    Mat wBar2 = w1.clone();
    wBar.addItem(wBar1);
    wBar.addItem(wBar2);
    w.addItem(w1);
    w.addItem(w2);
    Mat dx = w1.clone();
    Mat dy = w1.clone();
    p.addItem(dx);
    p.addItem(dy);
    Mat dxx = w1.clone();
    Mat dxy = w1.clone();
    Mat dyx = w1.clone();
    Mat dyy = w1.clone();
    q.addItem(dxx);
    q.addItem(dxy);
    q.addItem(dyx);
    q.addItem(dyy);

    Mat uGray;
    for (int i = 0; i < loopTimes; i++)
    {
        mat_vector u_bar_grad = derivativeForward(uBar);
        p = F_STAR_OPERATOR(p + (u_bar_grad - wBar) * sigma_n, alpha_u);

        mat_vector w_bar_second_derivative = symmetrizedSecondDerivative(wBar);
        q = F_STAR_OPERATOR(q + w_bar_second_derivative * sigma_n, alpha_w);

        Mat u_old = u.clone();
//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        Mat p_div = divergence(p);
        u = G_OPERATOR(u0, u + p_div*to_u, to_u, lambda);
        uBar = u + (u - u_old) * theta_n;

        mat_vector w_old = w.clone();
        mat_vector q_second_div = second_order_divergence(q);
        w = w + (p + q_second_div) * to_w;
        wBar = w + (w - w_old) * theta_w;

        double energe = GetEnerge(u, u0, w, edgeGrad, lambda, alpha_u, alpha_w);
        namedWindow("depthInpaint");
        double scale = 1./scaleDep;
        uGray = (u+0.5)*scale + minDep;
        uGray *=100;
        imshow("depthInpaint", uGray);
        waitKey(10);
        cout << "iter:" << i << " loss: " << energe << endl;
    }
    return uGray;
}