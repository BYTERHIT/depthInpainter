//
// Created by laiwenjie on 2021/10/28.
//

#ifndef DEPTHINPAINTER_SLIC_H
#define DEPTHINPAINTER_SLIC_H
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
typedef struct node {
    int x, y;
    double l, a, b;
}point;

//lab颜色空间像素
typedef struct labcolor {
    float l, a, b;
    labcolor(float x, float y, float z) {
        l = x;
        a = y;
        b = z;
    }
}labColor;

typedef struct colorinf {
    double r, g, b;
}RGBInfo;

typedef struct sprpxso {
    double r, g, b;
    int cnt, x, y, label;
}SuperPixel;


class SLIC {
private:
//初始间距
    int S;
//初始超像素个数
    const int nsp = 256;

//距离权重
    const double m = 40;

//迭代次数
    const int epoch = 10;

//记录当前像素到聚类中心的距离

    std::vector<std::vector<double>> dis;

//记录当前像素对应的聚类中心
    std::vector<std::vector<int>> cluster;

//记录所有聚类中心所包含的像素个数
    std::vector<int> clustercnt;

//聚类中心
    std::vector<point> centers;

    std::vector<SuperPixel> SuperPixels;
    std::vector<SuperPixel> KMcenters;
public:
    SLIC(cv::Mat img, int slics=256);
    ~SLIC();
    void InitImage(std::vector<std::vector<labColor>>& img, cv::Mat grad);
    void InitImage(cv::Mat cieImg, cv::Mat grad);
    void GenerateSuperpixel(std::vector<std::vector<labColor>>& img, int S);
    void DrawSuperPixel(cv::Mat img);
    void EnforceConnectivity(std::vector<std::vector<labColor>>& img);
    void UpdateCenters(cv::Mat img);
    void ReplacePixelColour(cv::Mat img);

};


#endif //DEPTHINPAINTER_SLIC_H
