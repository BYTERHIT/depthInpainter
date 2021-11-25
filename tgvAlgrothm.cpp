//
// Created by bytelai on 2021/11/3.
//

#include "tgvAlgrothm.h"
#include "SparseOptimizer.h"
#include "colorize.h"
#include <Eigen/SparseCore>
using namespace cv;
using namespace std;
using namespace Eigen;
#define MAX_LOOP_TIMES 1000000

Mat tgv_colorizeFiedility(Mat spImg, Mat rgb, Mat depth, Mat depthInit, TGV_PARAM param)
{
    Mat grayImg;
    cvtColor(rgb, grayImg,COLOR_BGR2GRAY);

    double minDep, maxDep;
    minMaxLoc(depth,&minDep,&maxDep);
    double scaleDep = 1. / maxDep;
    Mat depNorm = depth * scaleDep;
    Mat depInit = depthInit * scaleDep;
    threshold(depInit, depInit, 0,0,THRESH_TOZERO);
    threshold(depInit, depInit, 1,1,THRESH_TRUNC);

    double alpha_u = param.alpha_u, alpha_w = param.alpha_w;
    double delta = 0.05;
    double L = 12.;
    double mu = 2 * sqrt(param.lambda * delta) / L;
    double to_u = mu / (2 * param.lambda);
    double sigma_p = mu / (2*delta);
    double theta_u = 1 / (1 + mu);

    double lambda = 1. / param.lambda;//40;// 1 / lambda_tv;
    double sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_u;
    //初始化步长值 通过precontion算法实现


    SparseOptimizer spOp(rgb,depth,to_u, lambda,1.,1.);
//    double lambda = 40;// 1 / lambda_tv;
    int loopTimes = param.iterTimes;
    Mat zeros = Mat::zeros(depNorm.rows, depNorm.cols, depNorm.type());
    mat_vector w(2,zeros);
    mat_vector wBar(2,zeros);
    mat_vector p(2,zeros);
    mat_vector q(4,zeros);
    Mat u0 = depNorm.clone();
    Mat u, uBar;
    u =  u0.clone();
    uBar = u.clone();
    Mat u_old = u.clone();
    double tol = param.tol;
    if(tol>0)
        loopTimes = MAX_LOOP_TIMES;
    namedWindow("u0");
    imshow("u0", u0);
    namedWindow("depinit");
    imshow("depinit", depInit);


    for (int i = 0; i < loopTimes; i++)
    {
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif

#ifndef d_u_w
        p = F_STAR_OPERATOR(p + sigma / eta_p * alpha_u * (D_OPERATOR(tensor,u_bar_grad) - wBar), 1.);
#else
        mat_vector uBarGradMinusWBar = u_bar_grad - wBar;
        p = p + alpha_u * sigma_p * uBarGradMinusWBar;
        p = F_STAR_OPERATOR(p, 1.);
#endif
#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif

        q = F_STAR_OPERATOR(q + sigma_q * alpha_w * w_bar_second_derivative, 1.);

        uBar = u.clone();
        wBar = w.clone();

#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(p);
#else
        Mat p_div = divergenceForward(p);
#endif

        u = spOp.GetNewDepth(uBar + alpha_u * p_div * to_u);
//        u = G_OPERATOR(u0, uBar + alpha_u * p_div.mul(tau*to_u), tau * to_u, lambda, 0.);

#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif

        //此处和文献不一样p old?
#ifndef d_u_w
        w = wBar + (p*alpha_u + alpha_w * q_second_div).mul(tau * to_w);
#else
        w = wBar + (p*alpha_u + alpha_w * q_second_div)*to_w;
#endif

        uBar = u + (u - uBar) * theta_u;
        wBar = w + (w - wBar) * theta_w;
//        sigma = sigma / theta_n;
//        tau = tau * theta_n;


        if(i%10==0)
        {
            double tol_ =  sum(abs(u-u_old))[0];
            double tgv = GetTgvCost(u,w,mat_vector(),alpha_u, alpha_w);
            double fidelity = spOp.GetFidelity(u,lambda);
            namedWindow("depthInpaint");
            imshow("depthInpaint", u);
            waitKey(10);
            cout << "iter:" << i << " tgv: " << tgv << "fidelity: "<< fidelity << " tol: "<<tol_ << endl;
            if(tol_ < tol)
                break;
        }
        u_old = u.clone();
    }
    double scale = 1./scaleDep;
    u = (u)*scale;
    return u;
}

Mat tgv_colorizeFTGVL2(Mat spImg, Mat rgb, Mat depth,Mat depthInit, TGV_PARAM param) {
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(depth);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depNorm = mmNorm.norm;
    Mat depInit = (depthInit - minDep) * scaleDep;
    threshold(depInit, depInit, 0,0,THRESH_TOZERO);
    threshold(depInit, depInit, 1,1,THRESH_TRUNC);

    double theta_n = 1 ;
//    double alpha_u = 1., alpha_w = 2.;
    double alpha_u = param.alpha_u, alpha_w = param.alpha_w;
    Mat grayImg;
    cvtColor(rgb,grayImg,COLOR_BGR2GRAY);

    double tau = 0.1,sigma = 1./tau;
    double eta_p = 3.;
    double eta_q = 2.;
    double eta_r = 9.;
    //初始化步长值 通过precontion算法实现
    mat_vector tensor;
    if(spImg.empty())
        tensor = GetDGradMtx(grayImg,param.gama,param.gama);
    else
        tensor = GetTensor(spImg,grayImg);
    Mat a = tensor[0];
    Mat b = tensor[1];
    Mat c = tensor[2];
    Mat a2 = a.mul(a);
    Mat b2 = b.mul(b);
    Mat c2 = c.mul(c);
    Mat a_b2 = (a + c).mul(a + c);
    Mat b_c2 = (b + c).mul(b + c);
    Mat to_u = (a2 + b2 + 2 * c2 + a_b2 + b_c2)*(1.*1.);
    SYS_SPMTX sysSpmtx = GetSysSpMtx(rgb,Mat(),depNorm,1.,1);
    SparseMatrix<double> A = sysSpmtx.A;
    SparseMatrix<double> A_t = A.transpose();

    for (int k=0; k<A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator iter(A, k); iter; ++iter) {
            double Kij = abs(iter.value()); // 元素值
            int i = iter.row();   // 行标row index
            int j = iter.col();   // 列标（此处等于k）
            double *toPtr = (double *)to_u.data;
            *(toPtr + j) += pow(Kij,2-0.);
        }
    }


    Mat one = Mat::ones(grayImg.rows,grayImg.cols,CV_64FC1);
    divide(one,to_u,to_u);
    mat_vector to_w;
    Mat to_w1 = pow(1., 2) *(b.mul(b) + c.mul(c)) + 4 * pow(1.,2);
    divide(one, to_w1, to_w1);
    Mat to_w2 = pow(1., 2) *(a.mul(a) + c.mul(c)) + 4 * pow(1., 2);
    divide(one, to_w2, to_w2);
    to_w.addItem(to_w1);
    to_w.addItem(to_w2);

    double lambda = 1/ param.lambda;//40;// 1 / lambda_tv;
//    double lambda = 40;// 1 / lambda_tv;
    int loopTimes = param.iterTimes;
    Mat zeros = Mat::zeros(depNorm.rows, depNorm.cols, depNorm.type());
    mat_vector w(2,zeros);
    mat_vector wBar(2,zeros);
    mat_vector p(2,zeros);
    mat_vector q(4,zeros);
    Mat u0 = depNorm.clone();
    Mat r = zeros.clone();
    Mat u, uBar;
    u =  depInit.clone();
    uBar = u.clone();
    Mat u_old = u.clone();
    double tol = param.tol;
    if(tol>0)
        loopTimes = MAX_LOOP_TIMES;


    namedWindow("u0");
    imshow("u0", u0);
    namedWindow("depinit");
    imshow("depinit", depInit);
    double den = 1 + param.lambda * 1/eta_r;


    for (int i = 0; i < loopTimes; i++)
    {
//        if(sigma < 1000)
//        {
//            theta_n = 1 / sqrt(1 + 0.7 * tau);
//        }
//        else
//        {
//            theta_n = 1;
//        }
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif

#ifndef d_u_w
        p = F_STAR_OPERATOR(p + sigma / eta_p * alpha_u * (D_OPERATOR(tensor,u_bar_grad) - wBar), 1.);
#else
        mat_vector uBarGradMinusWBar = u_bar_grad - wBar;
        mat_vector du_tensor = D_OPERATOR(tensor, uBarGradMinusWBar);
        p = p + sigma / eta_p * du_tensor;
        p = F_STAR_OPERATOR(p, alpha_u);
#endif

#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif
        q = F_STAR_OPERATOR(q + sigma / eta_q * w_bar_second_derivative, alpha_w);

        Mat tmp = r + sigma /eta_r *(MatMulSp(A,uBar) - u0);
        r = tmp /den;

        uBar = u.clone();
        wBar = w.clone();

//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        mat_vector dp = D_OPERATOR(tensor, p);
#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(dp);
#else
        Mat p_div = divergenceForward(dp);
#endif

        Mat AtR = MatMulSp(A_t,r);
        u = uBar + (p_div - AtR).mul(tau*to_u);
//        u = G_OPERATOR(u0, uBar + alpha_u * p_div.mul(tau*to_u), tau * to_u, lambda, 0.);
#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif
        //此处和文献不一样p old?
#ifndef d_u_w
        w = wBar + (p + q_second_div).mul(tau * to_w);
#else
        w = wBar + (dp+ q_second_div).mul(tau*to_w);
#endif

        uBar = u + (u - uBar) * theta_n;
        wBar = w + (w - wBar) * theta_n;
//        sigma = sigma / theta_n;
//        tau = tau * theta_n;


        if(i%10==0)
        {
            double tol_ =  sum(abs(u-u_old))[0];
            double energe = GetEnerge(u, u0, w, tensor, lambda, alpha_u, alpha_w);
            namedWindow("depthInpaint");
            imshow("depthInpaint", u);
            waitKey(10);
            cout << "iter:" << i << " loss: " << energe << " tol: "<<tol_ << endl;
            if(tol_ < tol)
                break;
        }
        u_old = u.clone();
    }
    Mat imgSave;
    u.convertTo(imgSave,CV_8UC1,255.);
    imwrite("tgvColorF.png",imgSave);
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;

}


Mat tgv_colorizeFPrecontion(Mat spImg, Mat rgb, Mat dep, Mat depthInit, TGV_PARAM param)
{
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(dep);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depth = mmNorm.norm;

    Mat depInit = (depthInit - minDep) * scaleDep;
    threshold(depInit, depInit, 0,0,THRESH_TOZERO);
    threshold(depInit, depInit, 1,1,THRESH_TRUNC);

    double theta_n = 1 ;
    double alpha_u = param.alpha_u, alpha_w = param.alpha_w;

    double tau = 1,sigma = 1./tau;
    mat_vector tensor;
    Mat grayImg;
    cvtColor(rgb,grayImg,COLOR_BGR2GRAY);
    if(spImg.empty())
        tensor = GetDGradMtx(grayImg,param.gama,param.beta);
    else
        tensor = GetTensor(spImg,grayImg);


    //初始化步长值 通过precontion算法实现
    double alpha_precondition = 2.;
    double alpha_c = 0.5;
    mat_vector steps = GetSteps(tensor,depth.rows, depth.cols, alpha_u,alpha_w,alpha_precondition);
    Mat to_u = steps[0];
    mat_vector to_w(2);
    copy(steps.begin()+1,steps.begin()+3,to_w.begin());
    mat_vector sigma_p(2);
    copy(steps.begin() + 3, steps.begin()+5,sigma_p.begin());
    mat_vector sigma_q(4);
    copy(steps.begin()+5,steps.end(),sigma_q.begin());

    SYS_SPMTX sysSpmtx = GetSysSpMtx(rgb,Mat(),depth,alpha_c,1.);
    SparseMatrix<double> A =  1/sqrt(param.lambda)*sysSpmtx.A;
    SparseMatrix<double> A_t = A.transpose();
    Mat sigma_r = Mat::zeros(to_u.rows,to_u.cols,to_u.type());
    Mat to_u_inv;
    Mat one = Mat::ones(sigma_r.rows, sigma_r.cols, sigma_r.type());
    divide(one,to_u,to_u_inv);
    double *toPtr = (double *)to_u_inv.data;
    double *sigmaPtr = (double *)sigma_r.data;

    for (int k=0; k<A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            double Kij = abs(it.value()); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            *(toPtr + j) += pow(Kij,2-alpha_precondition);
            *(sigmaPtr + i) += pow(Kij,alpha_precondition);
        }
    }
    divide(one,to_u_inv,to_u);
    divide(one, sigma_r, sigma_r);

//    double lambda = 1/lambda_tv;
    double lambda = 1/param.lambda;
    int loopTimes = param.iterTimes;
    Mat zeros = Mat::zeros(depth.rows, depth.cols, depth.type());
    mat_vector w(2,zeros);
    mat_vector wBar(2,zeros);
    mat_vector p(2,zeros);
    mat_vector q(4,zeros);
    Mat r = zeros.clone();
    Mat u0 = depth.clone();
    Mat u, uBar;
    u =  depInit.clone();
    uBar = u.clone();
    Mat u_old = u.clone();

    namedWindow("u0");
    imshow("u0", u0);
    namedWindow("depinit");
    imshow("depinit", depInit);
    Mat f0 = sigma_r.mul(alpha_c/sqrt(param.lambda)*u0);
    double tol = param.tol;
    Mat Den = 1 + sigma_r;

    if(tol >0 )
    {
        loopTimes = MAX_LOOP_TIMES;
    }
    for (int i = 0; i < loopTimes; i++)
    {
//        if(sigma < 1000)
//        {
//            theta_n = 1 / sqrt(1 + 0.7 * tau);
//        }
//        else
//        {
//            theta_n = 1;
//        }
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + (D_OPERATOR(tensor,u_bar_grad) - wBar).mul(sigma_p*sigma), alpha_u);
#else
        p = F_STAR_OPERATOR(p + D_OPERATOR(tensor,u_bar_grad - wBar).mul(alpha_u*sigma*sigma_p), 1.);
#endif

#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif

        q = F_STAR_OPERATOR(q + w_bar_second_derivative.mul(alpha_w*sigma*sigma_q), 1.);


        Mat tmp = r + sigma * sigma_r.mul(MatMulSp(A,uBar)) - sigma*f0;
        divide(tmp,Den,r);


        uBar = u.clone();
        wBar = w.clone();

//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        mat_vector dp = D_OPERATOR(tensor, p);
#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(dp);
#else
        Mat p_div = divergenceForward(dp);
#endif

        Mat AtR = MatMulSp(A_t,r);

        u = uBar + (alpha_u*p_div-AtR).mul(tau*to_u);
//        u = G_OPERATOR(u0, uBar + p_div.mul(tau*to_u), to_u*tau, lambda, 0.);

#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif
        //此处和文献不一样p old?
#ifndef d_u_w
        w = wBar + (p*alpha_u + alpha_w * q_second_div).mul(to_w*tau);
#else
        w = wBar + (alpha_u*dp+ alpha_w*q_second_div).mul(tau*to_w);
#endif

        uBar = u + (u - uBar) * theta_n;
        wBar = w + (w - wBar) * theta_n;
//        sigma = sigma / theta_n;
//        tau = tau * theta_n;
        if(i%10==0)
        {
            double tol_ =  sum(abs(u-u_old))[0];
            double energe = GetEnerge(u, u0, w, tensor, lambda, alpha_u, alpha_w);
            namedWindow("u");
            imshow("u", u);
            waitKey(10);
            cout << "iter:" << i << " loss: " << energe << " tol: "<<tol_ << endl;
            if(tol_ <= param.tol)
                break;
        }
        u_old = u.clone();
    }
    Mat imgSave;
    u.convertTo(imgSave,CV_8UC1,255.);
    imwrite("colorizeFPrecodition.png",imgSave);
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;

}

/*
 *    [1] David Ferstl, Christian Reinbacher, Rene Ranftl, Matthias R眉ther
 *     and Horst Bischof, Image Guided Depth Upsampling using Anisotropic
 *     Total Generalized Variation, ICCV 2013.
 */
Mat tgv_algTGVL2(Mat spImg, Mat grayImg, Mat depth,Mat depthInit, Mat confidenceMap, TGV_PARAM param) {
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(depth);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depNorm = mmNorm.norm;
    Mat depInit = (depthInit - minDep) * scaleDep;
    threshold(depInit, depInit, 0,0,THRESH_TOZERO);
    threshold(depInit, depInit, 1,1,THRESH_TRUNC);

    double maxConf =0, minConf = 0;
    Mat confMap;
    if(!confidenceMap.empty())
    {
        minMaxLoc(confidenceMap,&minConf,&maxConf);
        confMap = (confidenceMap - minConf)/(maxConf - minConf);
    }
    else
    {
        threshold(depthInit, confMap, DBL_EPSILON,1,THRESH_BINARY);
    }
    confMap.convertTo(confMap,CV_64FC1);

    double theta_n = 1 ;
//    double alpha_u = 1., alpha_w = 2.;
    double alpha_u = param.alpha_u, alpha_w = param.alpha_w;

    double tau = 1,sigma = 1./tau;
    double eta_p = 3.;
    double eta_q = 2.;
    //初始化步长值 通过precontion算法实现
    mat_vector tensor;
    if(spImg.empty())
        tensor = GetDGradMtx(grayImg,param.gama,param.gama);
    else
        tensor = GetTensor(spImg,grayImg);
    Mat a = tensor[0];
    Mat b = tensor[1];
    Mat c = tensor[2];
    Mat a2 = a.mul(a);
    Mat b2 = b.mul(b);
    Mat c2 = c.mul(c);
    Mat a_b2 = (a + c).mul(a + c);
    Mat b_c2 = (b + c).mul(b + c);
    Mat to_u = (a2 + b2 + 2 * c2 + a_b2 + b_c2)*(alpha_u*alpha_u);
    Mat one = Mat::ones(grayImg.rows,grayImg.cols,CV_64FC1);
    divide(one,to_u,to_u);
    mat_vector to_w;
    Mat to_w1 = pow(alpha_u, 2) *(b.mul(b) + c.mul(c)) + 4 * pow(alpha_w,2);
    divide(one, to_w1, to_w1);
    Mat to_w2 = pow(alpha_u, 2) *(a.mul(a) + c.mul(c)) + 4 * pow(alpha_w, 2);
    divide(one, to_w2, to_w2);
    to_w.addItem(to_w1);
    to_w.addItem(to_w2);

    double lambda = 1/ param.lambda;//40;// 1 / lambda_tv;
//    double lambda = 40;// 1 / lambda_tv;
    int loopTimes = param.iterTimes;
    Mat zeros = Mat::zeros(depNorm.rows, depNorm.cols, depNorm.type());
    mat_vector w(2,zeros);
    mat_vector wBar(2,zeros);
    mat_vector p(2,zeros);
    mat_vector q(4,zeros);
    Mat u0 = depNorm.clone();
    Mat u, uBar;
    u =  depInit.clone();
    uBar = u.clone();
    Mat u_old = u.clone();
    double tol = param.tol;
    if(tol>0)
        loopTimes = MAX_LOOP_TIMES;


    namedWindow("u0");
    imshow("u0", u0);
    namedWindow("depinit");
    imshow("depinit", depInit);


    double gama = 1.;
    for (int i = 0; i < loopTimes; i++)
    {
//        if(sigma < 1000)
//        {
//            theta_n = 1 / sqrt(1 + gama * tau);
//        }
//        else
//        {
//            theta_n = 1;
//        }
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif

#ifndef d_u_w
        p = F_STAR_OPERATOR(p + sigma / eta_p * alpha_u * (D_OPERATOR(tensor,u_bar_grad) - wBar), 1.);
#else
        mat_vector uBarGradMinusWBar = u_bar_grad - wBar;
        mat_vector du_tensor = D_OPERATOR(tensor, uBarGradMinusWBar);
        p = p + alpha_u * sigma / eta_p * du_tensor;
        p = F_STAR_OPERATOR(p, 1.);
#endif
#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif

        q = F_STAR_OPERATOR(q + sigma / eta_q * alpha_w * w_bar_second_derivative, 1.);

        uBar = u.clone();
        wBar = w.clone();

//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        mat_vector dp = D_OPERATOR(tensor, p);
#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(dp);
#else
        Mat p_div = divergenceForward(dp);
#endif
        u = G_OPERATOR(u0, uBar + alpha_u * p_div.mul(tau*to_u), tau * to_u, lambda*confMap, 0.);

#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif
        //此处和文献不一样p old?
#ifndef d_u_w
        w = wBar + (p*alpha_u + alpha_w * q_second_div).mul(tau * to_w);
#else
        w = wBar + (dp*alpha_u + alpha_w * q_second_div).mul(tau*to_w);
#endif

        uBar = u + (u - uBar) * theta_n;
        wBar = w + (w - wBar) * theta_n;
//        sigma = sigma / theta_n;
//        tau = tau * theta_n;


        if(i%10==0)
        {
            double tol_ =  sum(abs(u-u_old))[0];
            double energe = GetEnerge(u, u0, w, tensor, lambda, alpha_u, alpha_w);
            namedWindow("depthInpaint");
            imshow("depthInpaint", u);
            waitKey(10);
            cout << "iter:" << i << " loss: " << energe << " tol: "<<tol_ << endl;
            if(tol_ < tol)
                break;
        }
        u_old = u.clone();
    }
    Mat imgSave;
    u.convertTo(imgSave,CV_8UC1,255.);
    imwrite("tgvl2.png",imgSave);
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;

}

Mat tgv_algPrecondition(Mat spImg, Mat grayImg, Mat dep,Mat depthInit, TGV_PARAM param) {

    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(dep);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depth = mmNorm.norm;

    Mat depInit = (depthInit - minDep) * scaleDep;
    threshold(depInit, depInit, 0,0,THRESH_TOZERO);
    threshold(depInit, depInit, 1,1,THRESH_TRUNC);

    double theta_n = 1 ;
    double alpha_u = param.alpha_u, alpha_w = param.alpha_w;

    double tau = 0.25,sigma = 1./tau;
    mat_vector tensor;
    if(spImg.empty())
        tensor = GetDGradMtx(grayImg,param.gama,param.beta);
    else
        tensor = GetTensor(spImg,grayImg,depth*255.);
    //初始化步长值 通过precontion算法实现
    mat_vector steps = GetSteps(tensor,depth.rows, depth.cols,alpha_u,alpha_w,1.);
    Mat to_u = steps[0];
    mat_vector to_w(2);
    copy(steps.begin()+1,steps.begin()+3,to_w.begin());
    mat_vector sigma_p(2);
    copy(steps.begin() + 3, steps.begin()+5,sigma_p.begin());
    mat_vector sigma_q(4);
    copy(steps.begin()+5,steps.end(),sigma_q.begin());

//    double lambda = 1/lambda_tv;
    double lambda = 1/param.lambda;
    int loopTimes = param.iterTimes;
    Mat zeros = Mat::zeros(depth.rows, depth.cols, depth.type());
    mat_vector w(2,zeros);
    mat_vector wBar(2,zeros);
    mat_vector p(2,zeros);
    mat_vector q(4,zeros);
    Mat u0 = depth.clone();
    Mat u, uBar;
    u =  depInit.clone();
    uBar = u.clone();
    Mat u_old = u.clone();

    namedWindow("u0");
    imshow("u0", u0);
    namedWindow("depinit");
    imshow("depinit", depInit);
    double tol = param.tol;

    if(tol >0 )
    {
        loopTimes = MAX_LOOP_TIMES;
    }
    for (int i = 0; i < loopTimes; i++)
    {
//        if(sigma < 3)
//        {
//            theta_n = 1 / sqrt(1 + 0.7 * tau);
//        }
//        else
//        {
//            theta_n = 1;
//        }
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + alpha_u * (D_OPERATOR(tensor,u_bar_grad) - wBar).mul(sigma_p*sigma), 1.);
#else
        p = F_STAR_OPERATOR(p + alpha_u * D_OPERATOR(tensor,u_bar_grad - wBar).mul(sigma*sigma_p), 1.);
#endif
#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif
        q = F_STAR_OPERATOR(q + alpha_w * w_bar_second_derivative.mul(sigma*sigma_q), 1.);

        uBar = u.clone();
        wBar = w.clone();
//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        mat_vector dp = D_OPERATOR(tensor, p);
#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(dp);
#else
        Mat p_div = divergenceForward(dp);
#endif

        u = G_OPERATOR(u0, uBar + alpha_u * p_div.mul(tau*to_u), to_u*tau, lambda, 0.);
#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif
        //此处和文献不一样p old?
#ifndef d_u_w
        w = wBar + (p*alpha_u + alpha_w * q_second_div).mul(to_w*tau);
#else
        w = wBar + (dp*alpha_u + alpha_w * q_second_div).mul(tau*to_w);
#endif

        uBar = u + (u - uBar) * theta_n;
        wBar = w + (w - wBar) * theta_n;
//        sigma = sigma / theta_n;
//        tau = tau * theta_n;
        if(i%10==0)
        {
            double tol_ =  sum(abs(u-u_old))[0];
            double energe = GetEnerge(u, u0, w, tensor, lambda, alpha_u, alpha_w);
            namedWindow("u");
            imshow("u", u);
            waitKey(10);
            cout << "iter:" << i << " loss: " << energe << " tol: "<<tol_ <<" tau:"<<tau<<" sigma: "<<sigma<<" theta: "<<theta_n<< endl;
            if(tol_ <= param.tol)
                break;
        }
        u_old = u.clone();
    }
    Mat imgSave;
    u.convertTo(imgSave,CV_8UC1,255.);
    imwrite("tgvPrecondition.png",imgSave);
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;
}


Mat tgv_alg2(vector<EDGE_GRAD> edgeGrad,Mat dep, int iterTimes, double lambda_tv, double L)
{
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(dep);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depth = mmNorm.norm;

    double alpha_u=1.,alpha_w=2.;
    double to_u=1./L, lambda = 1./lambda_tv, gama = lambda, sigma_p = 1./L/L/to_u,theta_u=1./sqrt(1+2*gama*to_u),sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_u;
    int loopTimes = iterTimes;
    Mat zeros = Mat::zeros(depth.rows, depth.cols, depth.type());
    mat_vector w(2, zeros);
    mat_vector wBar(2, zeros);
    mat_vector p(2, zeros);
    mat_vector q(4, zeros);
    Mat u0 = depth.clone();
    Mat u, uBar;
    u = zeros.clone();
    uBar = u0.clone();
    Mat uGray;

    for(int i = 0; i<loopTimes; i++)
    {
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + alpha_u * (D_OPERATOR(edgeGrad, u_bar_grad) - wBar)*(sigma_p), 1.);
#else
        p = F_STAR_OPERATOR(p + alpha_u * D_OPERATOR(edgeGrad, u_bar_grad - wBar)*(sigma_p), 1.);
#endif
#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif
        q = F_STAR_OPERATOR(q + alpha_w * w_bar_second_derivative*sigma_q,1.);

        Mat u_old = u.clone();
        mat_vector dp = D_OPERATOR(edgeGrad, p);
#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(dp);
#else
        Mat p_div = divergenceForward(dp);
#endif
        u = G_OPERATOR(u0,u + alpha_u * p_div * to_u,to_u,lambda);
        uBar = u + (u-u_old)*theta_u;

        mat_vector w_old = w.clone();
#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif

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
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;
}

Mat tgv_alg1(vector<EDGE_GRAD> edgeGrad, Mat dep,double lambda_tv, int n_it, double delta, double L)
{
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(dep);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depth = mmNorm.norm;

    double gamma = lambda_tv;
    double mu = 2 * sqrt(gamma * delta) / L;

    double to_u = mu / (2 * gamma);
    double sigma_n = mu / (2 * delta);

    double theta_n = 1 / (1 + mu);

    double alpha_u = 1., alpha_w = 2.;
    double lambda = 1/lambda_tv, gama = lambda;
    double sigma_p = sigma_n;//1. / L / L / to_u, theta_u = 1;// / sqrt(1 + 2 * gama * to_u)
    double sigma_q = sigma_p;
    double to_w = to_u, theta_w = theta_n;
    int loopTimes = n_it;
    Mat zeros = Mat::zeros(depth.rows, depth.cols, depth.type());
    mat_vector w(2, zeros);
    mat_vector wBar(2, zeros);
    mat_vector p(2, zeros);
    mat_vector q(4, zeros);
    Mat u0 = depth.clone();
    Mat u, uBar;
    u = zeros.clone();
    uBar = u0.clone();

    Mat uGray;
    for (int i = 0; i < loopTimes; i++)
    {
#ifdef USING_BACKWARD
        mat_vector u_bar_grad = derivativeBackward(uBar);
#else
        mat_vector u_bar_grad = derivativeForward(uBar);
#endif
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + alpha_u * (D_OPERATOR(edgeGrad,u_bar_grad) - wBar) * sigma_n, 1.);
#else
        p = F_STAR_OPERATOR(p + alpha_u * D_OPERATOR(edgeGrad,u_bar_grad - wBar) * sigma_n, 1.);
#endif

#ifdef USING_BACKWARD
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeBackward(wBar);
#else
        mat_vector w_bar_second_derivative = symmetrizedSecondDerivativeForward(wBar);
#endif
        q = F_STAR_OPERATOR(q + alpha_w * w_bar_second_derivative * sigma_n, 1.);

        Mat u_old = u.clone();
        mat_vector dp = D_OPERATOR(edgeGrad, p);
#ifdef USING_BACKWARD
        Mat p_div = divergenceBackward(dp);
#else
        Mat p_div = divergenceForward(dp);
#endif
        u = G_OPERATOR(u0, u + alpha_u * p_div*to_u, to_u, lambda);
        uBar = u + (u - u_old) * theta_n;

        mat_vector w_old = w.clone();
#ifdef USING_BACKWARD
        mat_vector q_second_div = secondOrderDivergenceBackward(q);
#else
        mat_vector q_second_div = secondOrderDivergenceForward(q);
#endif
#ifndef d_u_w
        w = w + (alpha_u * p + q_second_div) * to_w;
#else
        w = w + (alpha_u * dp + q_second_div) * to_w;
#endif
        wBar = w + (w - w_old) * theta_w;

        double energe = GetEnerge(u, u0, w, edgeGrad, lambda, alpha_u, alpha_w);
        namedWindow("depthInpaint");
        double scale = 1./scaleDep;
        uGray = u*scale + minDep;
        uGray *=50;
        uGray.convertTo(uGray,CV_8UC1);
        imshow("depthInpaint",uGray);
        waitKey(10);
        cout<<"iter:"<<i<<" loss: "<<energe<<endl;
    }
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;
}
