//
// Created by bytelai on 2021/11/3.
//

#include "tgvAlgrothm.h"
using namespace cv;
using namespace std;
//might be wrong
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

/*
 *    [1] David Ferstl, Christian Reinbacher, Rene Ranftl, Matthias R眉ther
 *     and Horst Bischof, Image Guided Depth Upsampling using Anisotropic
 *     Total Generalized Variation, ICCV 2013.
 */
Mat tgv_algTGVL2(Mat spImg, Mat grayImg, Mat depth, double lambda_tv, int n_it) {
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(depth);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depNorm = mmNorm.norm;

    double theta_n = 1 ;
//    double alpha_u = 1., alpha_w = 2.;
    double alpha_u = 1.2, alpha_w = 17.;

    double tau = 1.,sigma = 1./tau;
    double eta_p = 3.;
    double eta_q = 2.;
    //初始化步长值 通过precontion算法实现
    mat_vector tensor = GetTensor(spImg,grayImg);
    Mat a = tensor[0];
    Mat b = tensor[1];
    Mat c = tensor[2];
    Mat a2 = a.mul(a);
    Mat b2 = b.mul(b);
    Mat c2 = c.mul(c);
    Mat a_b2 = (a + c).mul(a + c);
    Mat b_c2 = (b + c).mul(b + c);
    Mat to_u = (a2 + b2 + 2 * c2 + a_b2 + b_c2)*(alpha_u*alpha_u);
    Mat one = Mat::ones(spImg.rows,spImg.cols,CV_64FC1);
    divide(one,to_u,to_u);
    mat_vector to_w;
    Mat to_w1 = pow(alpha_u, 2) *(b.mul(b) + c.mul(c)) + 4 * pow(alpha_w,2);
    divide(one, to_w1, to_w1);
    Mat to_w2 = pow(alpha_u, 2) *(a.mul(a) + c.mul(c)) + 4 * pow(alpha_w, 2);
    divide(one, to_w2, to_w2);
    to_w.addItem(to_w1);
    to_w.addItem(to_w2);

//    double lambda = 1/lambda_tv;//40;// 1 / lambda_tv;
    double lambda = 40;// 1 / lambda_tv;
    int loopTimes = n_it;
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

    for (int i = 0; i < loopTimes; i++)
    {
        if(sigma < 1000)
        {
            theta_n = 1 / sqrt(1 + 0.7 * tau);
        }
        else
        {
            theta_n = 1;
        }
        mat_vector u_bar_grad = derivativeForward(uBar);
        
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + sigma / eta_p * alpha_u * (D_OPERATOR(tensor,u_bar_grad) - wBar), 1.);
#else
        mat_vector uBarGradMinusWBar = u_bar_grad - wBar;
        mat_vector du_tensor = D_OPERATOR(tensor, uBarGradMinusWBar);
        p = p + alpha_u * sigma / eta_p * du_tensor;
        p = F_STAR_OPERATOR(p, 1.);
//        p = F_STAR_OPERATOR(p + sigma * alpha_u / eta_p * D_OPERATOR(tensor,u_bar_grad), 1.);
#endif

        mat_vector w_bar_second_derivative = symmetrizedSecondDerivative(wBar);
        q = F_STAR_OPERATOR(q + sigma / eta_q * alpha_w * w_bar_second_derivative, 1.);

        uBar = u.clone();
        wBar = w.clone();

//        mat_vector dp = p;// D_OPERATOR(edgeGrad, p);
        mat_vector dp = D_OPERATOR(tensor, p);
        Mat p_div = divergence(dp);
        u = G_OPERATOR(u0, uBar + alpha_u * p_div.mul(tau*to_u), tau * to_u, lambda, 0.);

        mat_vector q_second_div = second_order_divergence(q);
        //此处和文献不一样p old?
#ifndef d_u_w
        w = wBar + (p*alpha_u + alpha_w * q_second_div).mul(tau * to_w);
#else
        w = wBar + (dp*alpha_u + alpha_w * q_second_div).mul(tau*to_w);
#endif

        uBar = u + (u - uBar) * theta_n;
        wBar = w + (w - wBar) * theta_n;
        sigma = sigma / theta_n;
        tau = tau * theta_n;

        double tol =  sum(abs(u-u_old))[0];
        u_old = u.clone();

        double energe = GetEnerge(u, u0, w, tensor, lambda, alpha_u, alpha_w);
        namedWindow("depthInpaint");
        imshow("depthInpaint", u);
        waitKey(10);
        cout << "iter:" << i << " loss: " << energe << " tol: "<<tol << endl;
    }
    double scale = 1./scaleDep;
    u = (u)*scale + minDep;
    return u;

}

Mat tgv_algPrecondition(vector<EDGE_GRAD> edgeGrad, Mat dep, double lambda_tv = 0.03, int n_it = 1000) {
    MAX_MIN_NORM mmNorm = MaxMinNormalizeNoZero(dep);
    double minDep = mmNorm.min, maxDep = mmNorm.max;
    double scaleDep = 1. / (maxDep - minDep);
    Mat depth = mmNorm.norm;

    double theta_n = 1 ;
    double alpha_u = 1., alpha_w = 2.;

//    double tau = 1.,sigma = 1./tau;
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
//        if(sigma < 1000)
//        {
//            theta_n = 1 / sqrt(1 + 0.7 * tau);
//        }
//        else
//        {
//            theta_n = 1;
//        }
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
        u = G_OPERATOR(u0, u + alpha_u * p_div.mul(to_u), to_u, lambda, 0.);
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
//        sigma = sigma / theta_n;
//        tau = tau * theta_n;

        double tol =  sum(abs(u-u_old))[0];

        double energe = GetEnerge(u, u0, w, edgeGrad, lambda, alpha_u, alpha_w);
        namedWindow("depthInpaint");
        imshow("depthInpaint", u);
        waitKey(10);
        cout << "iter:" << i << " loss: " << energe << " tol: "<<tol << endl;
    }
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
    double to_u=1./L, lambda = 1./lambda_tv, gama = lambda, sigma_p = 1./L/L/to_u,theta_u=1/sqrt(1+2*gama*to_u),sigma_q = sigma_p;
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
        mat_vector u_bar_grad = derivativeForward(uBar);
#ifndef d_u_w
        p = F_STAR_OPERATOR(p + alpha_u * (D_OPERATOR(edgeGrad,u_bar_grad) - wBar) * sigma_n, 1.);
#else
        p = F_STAR_OPERATOR(p + alpha_u * D_OPERATOR(edgeGrad,u_bar_grad - wBar) * sigma_n, 1.);
#endif

        mat_vector w_bar_second_derivative = symmetrizedSecondDerivative(wBar);
        q = F_STAR_OPERATOR(q + alpha_w * w_bar_second_derivative * sigma_n, 1.);

        Mat u_old = u.clone();
        mat_vector dp = D_OPERATOR(edgeGrad, p);
        Mat p_div = divergence(dp);
        u = G_OPERATOR(u0, u + alpha_u * p_div*to_u, to_u, lambda);
        uBar = u + (u - u_old) * theta_n;

        mat_vector w_old = w.clone();
        mat_vector q_second_div = second_order_divergence(q);
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
