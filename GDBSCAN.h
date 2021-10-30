//
// Created by bytelai on 2021/10/29.
//

#ifndef DEPTHINPAINTER_GDBSCAN_H
#define DEPTHINPAINTER_GDBSCAN_H
#include "SLIC.h"
typedef struct
{
    SuperPixel sp;
    int refinedLabel = -1;
} SUPER_PIXEL;

typedef struct
{
    double r_2 = 0.;
    double b_2 = 0.;
    double g_2 = 0.;
    double d_2 = 0.;
    double x_2 = 0.;
    double y_2 = 0.;
    double rgbDis = 0.;
    double rgbdDis = 0.;
} DISTANCE;

class GDBSCAN {
public:
    GDBSCAN(std::vector<SuperPixel> sps, cv::Mat slicResult, float minDepPercent, float depthWeight, float xyWeight );
    ~GDBSCAN();
    std::vector<SUPER_PIXEL> GetSps();
    cv::Mat GetSpMap();
    void CLUSTER();
private:
    float _minDepPercent;//minCard
    float _simRgbdThresh = 5.;
    float _simRgbThresh = 2.;
    float _xyWeight = 1.;
    float _depWeight = 1.;
    std::vector<SUPER_PIXEL> _SPs;
    std::vector<DISTANCE> _distanceTable;
    cv::Mat _slicResult;
    std::vector<int> Find_RGBD_Neighbors(int spIdx);
    std::vector<int> Find_RGB_Neighbors(int spIdx);
    bool GDBSCAN::ExpandClusterUsingRgbDis(int spIdx);
    bool ExpandCluster(int spIdx,int clusterId);
    float GetDepthPercent(std::vector<int> sps);
    void ChangeClusterId(std::vector<int> pts, int id);
    bool IsNeighbor(SUPER_PIXEL sp1, SUPER_PIXEL sp2,double threshRgb = 3., double threshRgbd = 5.);
    DISTANCE GetDistanceFromTable(int i,int j);
    void CaculateDisTable();

};


#endif //DEPTHINPAINTER_GDBSCAN_H
