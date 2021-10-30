//
// Created by laiwenjie on 2021/10/28.
//

#include "SLIC.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <opencv2/opencv.hpp>
#include "GDBSCAN.h"
using namespace std;
using namespace cv;

double step8nbr[8][2] = { {-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0} };
double step4nbr[4][2] = { {-1,0},{0,1},{1,0},{0,-1} };



//xyz颜色空间像素
typedef struct xyzcolor {
    float x, y, z;
    xyzcolor(float a, float b, float c) {
        x = a;
        y = b;
        z = c;
    }
}xyzColor;

//rgb颜色空间像素
typedef struct rgbcolor {
    float r, g, b;
    rgbcolor(float x, float y, float z) {
        r = x;
        g = y;
        b = z;
    }
}rgbColor;


//rgb转xyz
xyzColor rgbToXyz(rgbColor c) {
    float x, y, z, r, g, b;

    r = c.r / 255.0; g = c.g / 255.0; b = c.b / 255.0;

    if (r > 0.04045)
        r = powf(((r + 0.055) / 1.055), 2.4);
    else r /= 12.92;

    if (g > 0.04045)
        g = powf(((g + 0.055) / 1.055), 2.4);
    else g /= 12.92;

    if (b > 0.04045)
        b = powf(((b + 0.055) / 1.055), 2.4);
    else b /= 12.92;

    r *= 100; g *= 100; b *= 100;

    x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    return xyzColor(x, y, z);
}

//xyz转lab
labColor xyzToCIELAB(xyzColor c) {
    float x, y, z, l, a, b;
    const float refX = 95.047, refY = 100.0, refZ = 108.883;

    x = c.x / refX; y = c.y / refY; z = c.z / refZ;

    if (x > 0.008856)
        x = powf(x, 1 / 3.0);
    else x = (7.787 * x) + (16.0 / 116.0);

    if (y > 0.008856)
        y = powf(y, 1 / 3.0);
    else y = (7.787 * y) + (16.0 / 116.0);

    if (z > 0.008856)
        z = powf(z, 1 / 3.0);
    else z = (7.787 * z) + (16.0 / 116.0);

    l = 116 * y - 16;
    a = 500 * (x - y);
    b = 200 * (y - z);

    return labColor(l, a, b);
}

//获取3*3区域中的最小梯度
void GetMinGrad(point& p, Mat img) {
    double mingrad = FLT_MAX;
    for (int i = p.y - 1; i <= p.y + 1; i++) {
        for (int j = p.x - 1; j <= p.x + 1; j++) {
            int right;
        }
    }
}

//计算梯度
void CalGrad(Mat grad, Mat intense) {
    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            double right, down, now = (double)intense.at<uchar>(i, j);
            if (j + 1 < grad.cols) right = (double)intense.at<uchar>(i, j + 1);
            else right = 0;
            if (i + 1 < grad.rows) down = (double)intense.at<uchar>(i + 1, j);
            else down = 0;
            grad.at<uchar>(i, j) = sqrt(pow(right - now, 2) + pow(down - now, 2));
        }
    }
}


void SLIC::InitImage(Mat cieImg, Mat grad)
{
    int height = cieImg.rows;
    int width = cieImg.cols;
    Mat cluster = Mat::ones(height,width,CV_32SC1) * -1;
    Mat dis = Mat::ones(height, width, CV_32FC1)*FLT_MAX;

    //在邻域内寻找最小梯度点
    for (int i = S / 2; i < height; i += S) {
        for (int j = S / 2; j < width; j += S) {
            point temp;
            double mingrad = FLT_MAX;
            for (int k = i - 1; k <= i + 1; k++) {
                for (int l = j - 1; l <= j + 1; l++) {
                    if (k < 0 || k >= height || l < 0 || l >= width) continue;
                    double now = (double)grad.at<uchar>(k, l);
                    if (now < mingrad) {
                        mingrad = now;
                        temp.x = l;
                        temp.y = k;
                    }
                }
            }

            temp.l = cieImg.at<Vec3f>(temp.y,temp.x)[0];
            temp.a = cieImg.at<Vec3f>(temp.y,temp.x)[1];
            temp.b = cieImg.at<Vec3f>(temp.y,temp.x)[2];

            centers.push_back(temp);//插入一个聚类中心
            clustercnt.push_back(0);//将该聚类中心包含的像素初始化为0
            //if (temp.x != i || temp.y != j) cnt++;
        }
    }
}
//初始化
void SLIC::InitImage(vector<vector<labColor>>& img, Mat grad) {
    //cols x     rows y
    int rows = img.size();
    int cols = img[0].size();

    for (int i = 0; i < rows; i++) {
        vector<int> clstrR;
        vector<double> disR;
        for (int j = 0; j < cols; j++) {
            clstrR.push_back(-1);
            disR.push_back(FLT_MAX);
        }
        dis.push_back(disR);
        cluster.push_back(clstrR);
    }


    //在邻域内寻找最小梯度点
    for (int i = S / 2; i < rows; i += S) {
        for (int j = S / 2; j < cols; j += S) {

            point temp;
            double mingrad = FLT_MAX;
            for (int k = i - 1; k <= i + 1; k++) {
                for (int l = j - 1; l <= j + 1; l++) {
                    if (k < 0 || k >= rows || l < 0 || l >= cols) continue;
                    double now = (double)grad.at<uchar>(k, l);
                    if (now < mingrad) {
                        mingrad = now;
                        temp.x = l;
                        temp.y = k;
                    }
                }
            }

            temp.l = img[temp.y][temp.x].l;
            temp.a = img[temp.y][temp.x].a;
            temp.b = img[temp.y][temp.x].b;

            centers.push_back(temp);//插入一个聚类中心
            clustercnt.push_back(0);//将该聚类中心包含的像素初始化为0
            //if (temp.x != i || temp.y != j) cnt++;
        }
    }
}


//计算像素间距离
double CalDistence(point x, point y, double S, double m) {
    double dc = sqrt(pow(x.l - y.l, 2) + pow(x.a - y.a, 2) + pow(x.b - y.b, 2));
    double ds = sqrt(pow(x.x - y.x, 2) + pow(x.y - y.y, 2));
    double D = sqrt(pow(dc, 2) + pow(ds / S, 2) * pow(m, 2));
    return D;
}


//计算迭代残差
double CalResError(vector<point>& pre, vector<point>& now) {
    double re = 0;
    for (int i = 0; i < pre.size(); i++) {
        re += sqrt(pow(pre[i].x - now[i].x, 2) + pow(pre[i].y - now[i].y, 2));
    }
    return re;
}


//生成超像素
void SLIC::GenerateSuperpixel(vector<vector<labColor>>& img, int S) {
    int rows = img.size();
    int cols = img[0].size();

    //迭代次数
    for (int iter = 0; iter < epoch; iter++) {
        //遍历所有聚类中心
        for (int t = 0; t < centers.size(); t++) {
            point center = centers[t];
            //2S*2S中的行
            for (int i = center.y - S; i <= center.y + S; i++) {
                //2S*2S中的列
                for (int j = center.x - S; j <= center.x + S; j++) {
                    if (i >= 0 && i < rows && j >= 0 && j < cols) {
                        point dst;
                        dst.x = j;
                        dst.y = i;
                        dst.l = img[i][j].l;
                        dst.a = img[i][j].a;
                        dst.b = img[i][j].b;
                        double D = CalDistence(center, dst,S,m);
                        if (D < dis[i][j]) {
                            dis[i][j] = D;
                            cluster[i][j] = t;
                        }
                    }
                }
            }
        }

        vector<point> pre = centers;

        for (int i = 0; i < centers.size(); i++) {
            centers[i].l = centers[i].a = centers[i].b = centers[i].x = centers[i].y = 0;
        }


        //计算新聚类中心
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int index = cluster[i][j];
                //cout << index << endl;
                if (index < 0) {
                    cout << "error" << endl;
                    cout << "row=" << i << " cols=" << j << endl;
                    continue;
                }
                centers[index].l += img[i][j].l;
                centers[index].a += img[i][j].a;
                centers[index].b += img[i][j].b;
                centers[index].x += j;
                centers[index].y += i;
                clustercnt[index]++;
            }
        }

        //正则化
        for (int i = 0; i < centers.size(); i++) {
            centers[i].l /= clustercnt[i];
            centers[i].a /= clustercnt[i];
            centers[i].b /= clustercnt[i];
            centers[i].x /= clustercnt[i];
            centers[i].y /= clustercnt[i];
        }


        double residualerror = CalResError(pre, centers);
        printf("epoch=%d,error=%lf\n", iter, residualerror);

    }
}

//绘制超像素边界
void SLIC::DrawSuperPixel(Mat img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            bool isEdge = false;
            for (int k = 0; k < 4; k++) {
                int y = i + step4nbr[k][0];
                int x = j + step4nbr[k][1];
                if (y >= 0 && y < img.rows && x >= 0 && x < img.cols
                    && cluster[i][j] != cluster[y][x]) {
                    isEdge = true;
                    break;
                }
            }
            if (isEdge) {
                Point p(j, i);
                circle(img, p, 0, Scalar(0, 0, 0), -1);
            }
        }
    }
}


//强制连续性
void SLIC::EnforceConnectivity(vector<vector<labColor>>& img) {
    int rows = img.size();
    int cols = img[0].size();

    int adjlabel = 0, label = 0;

    int threshold = rows * cols / centers.size();
    vector<vector<int>> newcluster;
    _clusterResult = Mat::zeros(rows,cols,CV_16UC1);

    for (int i = 0; i < rows; i++) {
        vector<int> newrows;
        for (int j = 0; j < cols; j++) {
            newrows.push_back(-1);
        }
        newcluster.push_back(newrows);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (newcluster[i][j] == -1) {
                newcluster[i][j] = label;
                _clusterResult.at<uint16_t>(i,j) = label;

                //BFS
                queue<pair<int, int>> q;   //BFS队列
                vector<pair<int, int>> connectiveregion;   //连通区域

                q.push(pair<int, int>(i, j));
                connectiveregion.push_back(pair<int, int>(i, j));

                //在四邻域中寻找新聚类中心
                for (int t = 0; t < 4; t++) {
                    int y = i + step4nbr[t][0];
                    int x = j + step4nbr[t][1];
                    if (y >= 0 && y < rows && x >= 0 && x < cols) {
                        if (newcluster[y][x] >= 0) {
                            adjlabel = newcluster[y][x];
                        }
                    }
                }

                int numclusterpixel = 1;
                while (!q.empty()) {
                    pair<int, int> now = q.front();
                    q.pop();
                    for (int k = 0; k < 4; k++) {
                        int y = now.first + step4nbr[k][0];
                        int x = now.second + step4nbr[k][1];
                        if (y >= 0 && y < rows && x >= 0 && x < cols
                            && newcluster[y][x] == -1 && cluster[y][x] == cluster[i][j]) {
                            numclusterpixel++;
                            q.push(pair<int, int>(y, x));
                            connectiveregion.push_back(pair<int, int>(y, x));
                            newcluster[y][x] = label;
                            _clusterResult.at<uint16_t>(y,x) = label;
                        }
                    }
                }

                //cout << connectiveregion.size() << endl;
                //cout << numclusterpixel << endl;
                //区域面积小于阈值，进行合并
                if (numclusterpixel <= threshold / 2) {
                    for (int k = 0; k < connectiveregion.size(); k++) {
                        newcluster[connectiveregion[k].first][connectiveregion[k].second] = adjlabel;
                        _clusterResult.at<uint16_t>(connectiveregion[k].first,connectiveregion[k].second) = label;
                    }
                    //cout << adjlabel << endl;
                    label--;
                }
                label++;
            }
        }
    }

    /*
    for (int i = 110; i < 150; i++) {
        for (int j = 280; j < 310; j++) {
            printf("%3d ", cluster[i][j]);
        }
        cout << endl;
    }
    */
    cluster = newcluster;
    //cout << "________________________________________________" << endl;
}


bool cmp(const SuperPixel& a, const SuperPixel& b) {
    return a.label < b.label;
}


//更新超像素中心信息
void SLIC::UpdateCenters(Mat img) {
    for (int i = 0; i < cluster.size(); i++) {
        for (int j = 0; j < cluster[i].size(); j++) {
            bool isFind = false;
            for (int k = 0; k < SuperPixels.size(); k++) {
                if (SuperPixels[k].label == cluster[i][j]) {
                    SuperPixels[k].b += img.at<cv::Vec3b>(i, j)[0];
                    SuperPixels[k].g += img.at<cv::Vec3b>(i, j)[1];
                    SuperPixels[k].r += img.at<cv::Vec3b>(i, j)[2];
                    SuperPixels[k].x += j;
                    SuperPixels[k].y += i;
                    SuperPixels[k].d += _depth.at<double>(i,j);
                    SuperPixels[k].dcnt += (_depth.at<double>(i,j) > 0);
                    SuperPixels[k].cnt++;
                    SuperPixels[k].pts.push_back(Point(j,i));
                    isFind = true;
                    break;
                }
            }
            if (!isFind) {
                SuperPixel temp;
                temp.b = img.at<cv::Vec3b>(i, j)[0];
                temp.g = img.at<cv::Vec3b>(i, j)[1];
                temp.r = img.at<cv::Vec3b>(i, j)[2];
                temp.d = _depth.at<double>(i,j);
                temp.dcnt = (_depth.at<double>(i,j) > 0);
                temp.x = j;
                temp.y = i;
                temp.cnt = 1;
                temp.label = cluster[i][j];
                temp.pts.push_back(Point(j,i));
                SuperPixels.push_back(temp);
            }
        }
    }

    for (int i = 0; i < SuperPixels.size(); i++) {
        SuperPixels[i].r /= SuperPixels[i].cnt;
        SuperPixels[i].g /= SuperPixels[i].cnt;
        SuperPixels[i].b /= SuperPixels[i].cnt;
        SuperPixels[i].d /= (SuperPixels[i].dcnt+FLT_EPSILON);
        SuperPixels[i].x /= SuperPixels[i].cnt;
        SuperPixels[i].y /= SuperPixels[i].cnt;
    }
    sort(SuperPixels.begin(), SuperPixels.end(), cmp);
}

//将每个超像素中包含的像素RGB替换为平均值
void SLIC::ReplacePixelColour(Mat img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img.at<cv::Vec3b>(i, j)[0] = (uchar)SuperPixels[cluster[i][j]].b;
            img.at<cv::Vec3b>(i, j)[1] = (uchar)SuperPixels[cluster[i][j]].g;
            img.at<cv::Vec3b>(i, j)[2] = (uchar)SuperPixels[cluster[i][j]].r;
        }
    }
}


SLIC::SLIC(cv::Mat src, Mat dep, int slics ){
    _srcImg = src.clone();
    nsp = slics;
    int N = src.cols * src.rows;
    S = (int)sqrt(N / nsp);
    //cout << "S=" << S << endl;
    Mat intense;
    _depth = dep.clone();
    cvtColor(src,intense,COLOR_BGR2GRAY);

    Mat grad = Mat::zeros(intense.size(), intense.type());
    CalGrad(grad, intense);

//    Mat cieLab;
//    cvtColor(src,cieLab,COLOR_BGR2Lab);
//TODO 整体可以用opencv 重构

    vector<vector<labColor>> cielab;
    for (int i = 0; i < src.rows; i++) {
        vector<labColor> row;
        for (int j = 0; j < src.cols; j++) {
            row.push_back(labColor(0, 0, 0));
        }
        cielab.push_back(row);
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float xr, xg, xb, yl, ya, yb;
            xr = src.at<cv::Vec3b>(i, j)[2];
            xg = src.at<cv::Vec3b>(i, j)[1];
            xb = src.at<cv::Vec3b>(i, j)[0];
            xyzColor temp = rgbToXyz(rgbColor(xr, xg, xb));
            labColor temp2 = xyzToCIELAB(temp);

            cielab[i][j].l = temp2.l;
            cielab[i][j].a = temp2.a;
            cielab[i][j].b = temp2.b;
        }
    }

    InitImage(cielab, grad);
    GenerateSuperpixel(cielab, S);
    for (int i = 0; i < 1; i++) {
        EnforceConnectivity(cielab);
    }
    //Mat convas = _srcImg.clone();
    //DrawSuperPixel(convas);

    //imwrite("SuperPixels.jpg", convas);


    Mat src2= src.clone();
    UpdateCenters(src2);
//    ReplacePixelColour(src2);
//    imwrite("SuperPixelSegment.jpg", src2);
}
void SLIC::GreedAggregateSPWithDelpth()
{
    GDBSCAN * greedAggresor = new GDBSCAN(SuperPixels, _clusterResult, 0.2, 650.25, pow(m/S/2,2));
    greedAggresor->CLUSTER();
    vector<SUPER_PIXEL> sps =greedAggresor->GetSps();
//    _segResult = Mat::zeros(cluster.size(),cluster[0].size(),CV_8UC1);
    _segResult = greedAggresor->GetSpMap();


//    for (int i = 0; i < cluster.size(); i++) {
//        for (int j = 0; j < cluster[i].size(); j++) {
//            bool isFind = false;
//            for (int k = 0; k < sps.size(); k++) {
//                if (sps[k].sp.label == cluster[i][j]) {
//                    _segResult.at<uchar>(i,j) = sps[k].refinedLabel;
//                    break;
//                }
//            }
//        }
//    }
    //Mat canvas = _srcImg;
    //DrawSuperPixelUsingMat(canvas);
    //imwrite("SuperPixelsAddDepth.jpg", canvas);

}

Mat SLIC::GetSpMap()
{
    return _segResult.clone();
}

void SLIC::DrawSuperPixelUsingMat(Mat img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            bool isEdge = false;
            for (int k = 0; k < 4; k++) {
                int y = i + step4nbr[k][0];
                int x = j + step4nbr[k][1];
                if (y >= 0 && y < img.rows && x >= 0 && x < img.cols
                    && _segResult.at<uchar>(i,j) != _segResult.at<uchar>(y,x)) {
                    isEdge = true;
                    break;
                }
            }
            if (isEdge) {
                Point p(j, i);
                circle(img, p, 0, Scalar(0, 0, 0), -1);
            }
        }
    }
}

