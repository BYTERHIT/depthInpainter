//
// Created by laiwenjie on 2021/10/26.
//

#include "tgvOperator.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "mat_vector.h"

using namespace cv;
using namespace std;
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
    yDiv(xRoi2)+=xy_yx(xRoi2)-xxGrad(xRoi1);
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
   if(vecSize==4)
   {
       Mat xx = pBar[0];
       Mat xy = pBar[1];
       Mat yx = pBar[2];
       Mat yy = pBar[3];
       Mat sum = abs(xx)+abs(xy) + abs(yx) + abs(yy);
//       Mat sum = sqrt(xx.mul(xx) + yx.mul(yx)+xy.mul(xy)+yy.mul(yy));
       int width = xx.cols, height = xx.rows;
       double *ptr = (double*)sum.data;
       for(int i = 0 ;i <width*height;i++)
       {
           if(*ptr < alpha)
               *ptr = 1.;
           ptr++;
       }
       divide(xx,sum,xx);
       divide(xy,sum,xy);
       divide(yx,sum,yx);
       divide(yy,sum,yy);
       result.addItem(xx);
       result.addItem(xy);
       result.addItem(yx);
       result.addItem(yy);
   }
   else if(vecSize==2)
   {
       Mat x = pBar[0];
       Mat y = pBar[1];
       Mat sum = abs(x)+abs(y);
//       Mat sum = sqrt(x.mul(x) + y.mul(y);
       int width = x.cols, height = x.rows;
       double *ptr = (double*)sum.data;
       for(int i = 0 ;i <width*height;i++)
       {
           if(*ptr < alpha)
               *ptr = 1.;
           ptr++;
       }
       divide(x,sum,x);
       divide(y,sum,y);
       result.addItem(x);
       result.addItem(y);
   }
   return result;
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
        if(*gPtr)
            *uPtr = (*uBarPtr + to * lambda * (*gPtr))/(1. + to*lambda);
        else
            *uPtr = (*uBarPtr + to * lambda * (*gPtr));
        uPtr++;
        uBarPtr++;
        gPtr++;
    }
    return u;
}

Mat tgv_alg3(vector<EDGE_GRAD> edgeGrad,Mat depth)
{
    double L = 24.0,alpha0=0.05,alpha1=0.05;
    double to_u=1./L, lambda = 10., gama = lambda,
    delta = alpha0, mu = 2* sqrt(gama*delta)/L,
    tau_n = mu / (2 * gama), sigma_n = mu / (2*delta), theta_n = 1 / (1 + mu),
    sigma_p = 1./L/L/to_u,theta_u=1/sqrt(1+2*gama*to_u),sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_u;
    int loopTimes = 1000;
    mat_vector w,p,q,wBar;
    Mat u0 = depth.clone();
    Mat u,uBar;
    Mat w1 = Mat::zeros(depth.rows,depth.cols, depth.type());
    Mat w2 = w1.clone();
    u = w1.clone();
    uBar = w1.clone();
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
        p= F_STAR_OPERATOR(p,1.);
        Mat u_old = u.clone();
        mat_vector w_old = w.clone();

        q = q + symmetrizedSecondDerivative(wBar)*sigma_n;
        q = F_STAR_OPERATOR(q,1.);

        mat_vector dp = D_OPERATOR(edgeGrad,p);
        Mat uDelta = divergence(dp) * tau_n;
        u = u + uDelta;
        u = G_OPERATOR(u0,u,tau_n,lambda);

        theta_n = 1 / sqrt( 1 + 2 * gama * tau_n);
        tau_n = theta_n * tau_n;
        sigma_n = sigma_n / theta_n;

        w = second_order_divergence(q) + p;
        w = w + w * tau_n;

        uBar = u + (u-u_old)*theta_n;
        wBar = w + (w-w_old)*theta_n;
        namedWindow("depthInpaint");
        Mat uGray = u * 256.;
        uGray.convertTo(uGray,CV_8UC1);
        imshow("depthInpaint",uGray);
        waitKey(10);
        cout<<"iter:"<<i<<endl;
    }
    return u;
}

Mat tgv_alg2(vector<EDGE_GRAD> edgeGrad,Mat depth)
{
    double L = 8.0,alpha0=1.,alpha1=1.;
    double to_u=1./L, lambda = 16., gama = lambda, sigma_p = 1./L/L/to_u,theta_u=1/sqrt(1+2*gama*to_u),sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_u;
    int loopTimes = 1000;
    mat_vector w,p,q,wBar;
    Mat u0 = depth.clone();
    Mat u,uBar;
    Mat w1 = Mat::zeros(depth.rows,depth.cols, depth.type());
    Mat w2 = w1.clone();
    u = w1.clone();
    uBar = w1.clone();
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
        p = p+du*sigma_p;
        p= F_STAR_OPERATOR(p,alpha1);

        Mat u_old = u.clone();
        mat_vector w_old = w.clone();

        q = q + symmetrizedSecondDerivative(wBar)*sigma_q;
        q = F_STAR_OPERATOR(q,alpha0);

        mat_vector dp = D_OPERATOR(edgeGrad,p);
        Mat uDelta = divergence(dp) * to_u;
        u = u + uDelta;
        u = G_OPERATOR(u0,u,to_u,lambda);

        w = second_order_divergence(q) + p;
        w = w + w * to_w;

        theta_u = 1/sqrt(1+2*gama*to_u);
        to_u = theta_u*to_u;
        sigma_p = sigma_p / theta_u;

        theta_w = 1/sqrt(1 + 2 * gama*to_w);
        to_w = theta_w* to_w;
        sigma_q = sigma_q / theta_w;

//        uBar = u + (u-uBar)*theta_u;
//        wBar = w + (w-wBar)*theta_w;
        uBar = u + (u-u_old)*theta_u;
        wBar = w + (w-w_old)*theta_w;
        namedWindow("depthInpaint");
        Mat uGray = u * 256.;
        uGray.convertTo(uGray,CV_8UC1);
        imshow("depthInpaint",uGray);
        waitKey(10);
        cout<<"iter:"<<i<<endl;
    }
    return u;
}