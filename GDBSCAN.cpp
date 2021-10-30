//
// Created by bytelai on 2021/10/29.
//

#include "GDBSCAN.h"
#include <iostream> 
#include <fstream>
using namespace std;
using namespace cv;

#define UNCLASSIFIED_ID -1
#define NOISE_ID -2
void GDBSCAN::CaculateDisTable() {
    int size = _SPs.size();
    DISTANCE dis;
    ofstream myfile;
    myfile.open("disTable.txt",ios::out);
    for(int i = 0; i<size; i++)
    {
        SUPER_PIXEL sp1 = _SPs[i];
        for(int j = i; j < size; j++)
        {
            SUPER_PIXEL sp2 = _SPs[j];
            dis.r_2 = pow(sp1.sp.r - sp2.sp.r,2);
            dis.b_2 = pow(sp1.sp.b - sp2.sp.b,2);
            dis.g_2 = pow(sp1.sp.g - sp2.sp.g,2);
            if(sp1.sp.dcnt == 0 || sp2.sp.dcnt == 0)
                dis.d_2 = 0.;
            else
                dis.d_2 = pow(sp1.sp.d - sp2.sp.d,2)*_depWeight;
            dis.x_2 = abs(sp1.sp.x - sp2.sp.x);
            dis.y_2 = abs(sp1.sp.y - sp2.sp.y);
            dis.rgbDis = sqrt((dis.r_2 + dis.b_2 + dis.g_2) / (3));
            dis.rgbdDis = sqrt((dis.r_2 + dis.b_2 + dis.g_2 + dis.d_2) / (4));
            _distanceTable.push_back(dis);
            myfile << sp1.sp.label << "->"<< sp2.sp.label<<": r_2: " << dis.r_2 << "\t b_2: " << dis.b_2 << "\t g_2: " << dis.g_2 << "\t d_2: " << dis.d_2 << "\t x_2: " << dis.x_2 << "\t y_2: " << dis.y_2 << "\t rgbDis: " << dis.rgbDis << "\t rgbdDis: "<<dis.rgbdDis<< endl;
        }
    }
    myfile.close();
}

DISTANCE GDBSCAN::GetDistanceFromTable(int i,int j)
{
    int index0 = min(i,j);
    int index1 = max(i,j);
    int index = index0*_SPs.size()  - (index0) * (index0+1) / 2 + index1;
    DISTANCE dis = _distanceTable[index];
    return dis;
}

bool GDBSCAN::IsNeighbor(SUPER_PIXEL sp1, SUPER_PIXEL sp2, double threshRgb, double threshRgbd)
{
    DISTANCE dis = GetDistanceFromTable(sp1.sp.label,sp2.sp.label);
    bool ret = false;
    if(sp1.sp.dcnt == 0 || sp2.sp.dcnt == 0) {
        if (dis.rgbDis < threshRgb) {
            ret = true;
        }
    }else
    {
        if (dis.rgbdDis < threshRgbd) {
            ret = true;
        }
    }
//    if(ret)
//        cout << sp1.sp.label << "->"<< sp2.sp.label<<": r_2: " << dis.r_2 << "\t b_2: " << dis.b_2 << "\t g_2: " << dis.g_2 << "\t d_2: " << dis.d_2 << "\t x_2: " << dis.x_2 << "\t y_2: " << dis.y_2 << "\t rgbDis: " << dis.rgbDis << "\t rgbdDis: "<<dis.rgbdDis<< endl;
    return ret;
}
//w_card
float GDBSCAN::GetDepthPercent(std::vector<int> sps) {
    float percent = 0;
    float total = 0;
    for(auto iter = sps.begin(); iter != sps.end(); iter++)
    {
        percent += _SPs[*iter].sp.dcnt;
        total += _SPs[*iter].sp.cnt;
    }
    return percent / total;
}

GDBSCAN::GDBSCAN(std::vector<SuperPixel> sps, cv::Mat slicResult, float minDepPercent, float depthWeight, float xyWeight ) {
    for(auto iter = sps.begin(); iter!=sps.end();iter++)
    {
        SUPER_PIXEL sp;
        sp.sp = *iter;
        sp.refinedLabel = UNCLASSIFIED_ID;
        _SPs.push_back(sp);
    }
    _xyWeight = xyWeight;
    _depWeight = depthWeight;
    _minDepPercent = minDepPercent;
    _slicResult = slicResult;
    CaculateDisTable();
}
vector<int> GDBSCAN::Find_RGBD_Neighbors(int spIdx) {
    vector<int> ret;
    double disMin = FLT_MAX;
    int minIdx = 0;
    int idx = 0;
    SUPER_PIXEL sp1 = _SPs[spIdx];
    for(auto iter = _SPs.begin(); iter!=_SPs.end(); iter++) {
        SUPER_PIXEL sp2 = *iter;
        DISTANCE dis = GetDistanceFromTable(sp1.sp.label, sp2.sp.label);
        bool isNeighbor = false;
        if (sp1.sp.dcnt != 0 && sp2.sp.dcnt != 0 && dis.rgbdDis < _simRgbdThresh) {
            {
                isNeighbor = true;
            }
            if (isNeighbor) {
//                cout << sp1.sp.label << "->" << sp2.sp.label << ": r_2: " << dis.r_2 << "\t b_2: " << dis.b_2
//                     << "\t g_2: " << dis.g_2 << "\t d_2: " << dis.d_2 << "\t x_2: " << dis.x_2 << "\t y_2: " << dis.y_2
//                     << "\t rgbDis: " << dis.rgbDis << "\t rgbdDis: " << dis.rgbdDis << endl;
                ret.push_back(sp2.sp.label);
            }
        }
    }
    return ret;
}
vector<int> GDBSCAN::Find_RGB_Neighbors(int spIdx){
    vector<int> neighbors;
    SUPER_PIXEL sp1 = _SPs[spIdx];
    for(auto iter = _SPs.begin(); iter!=_SPs.end(); iter++)
    {
        SUPER_PIXEL sp2 = *iter;
        DISTANCE dis = GetDistanceFromTable(sp1.sp.label,sp2.sp.label);
        bool isNeighbor = false;
        if(sp2.refinedLabel < 0 && dis.rgbDis < _simRgbThresh) {
            isNeighbor = true;
        }
        if (isNeighbor)
        {
//            cout << sp1.sp.label << "->" << sp2.sp.label << ": r_2: " << dis.r_2 << "\t b_2: " << dis.b_2 << "\t g_2: " << dis.g_2 << "\t d_2: " << dis.d_2 << "\t x_2: " << dis.x_2 << "\t y_2: " << dis.y_2 << "\t rgbDis: " << dis.rgbDis << "\t rgbdDis: " << dis.rgbdDis << endl;
            neighbors.push_back(sp2.sp.label);
        }
    }
    return neighbors;
}
bool GDBSCAN::ExpandClusterUsingRgbDis(int spIdx) {
    double disMin = FLT_MAX;
    int minIdx = 0;
    int idx = 0;
    SUPER_PIXEL sp1 = _SPs[spIdx];
    for(auto iter = _SPs.begin(); iter!=_SPs.end(); iter++)
    {
        SUPER_PIXEL sp2 = *iter;
        DISTANCE dis = GetDistanceFromTable(sp1.sp.label,sp2.sp.label);
        if(sp2.refinedLabel >= 0 && dis.rgbDis < disMin) {
            disMin = dis.rgbDis;
            minIdx = sp2.sp.label;
        }
    }
    int clusterId = minIdx;
    if(disMin < _simRgbThresh) {
        vector<int> neighbors = Find_RGB_Neighbors(spIdx);
        ChangeClusterId(neighbors, clusterId);
        neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), spIdx),neighbors.end());
        while(!neighbors.empty())
        {
            int currentPt = neighbors[0];
            vector<int> result = Find_RGB_Neighbors(currentPt);
            for (auto iter = result.begin(); iter != result.end(); iter++) {
                neighbors.push_back(*iter);
                ChangeClusterId({*iter},clusterId);
            }
            neighbors.erase(remove(neighbors.begin(),neighbors.end(),currentPt));
        }
        return true;
    }
    return false;
}

void GDBSCAN::ChangeClusterId(std::vector<int> pts, int id) {
    for(auto iter = pts.begin(); iter!=pts.end();iter++)
    {
        _SPs[*iter].refinedLabel = id;
        auto pts =  _SPs[*iter].sp.pts;
        for(auto iterIner = pts.begin() ;iterIner != pts.end(); iterIner ++)
        {
            _slicResult.at<uint16_t>(*iterIner) = id;
        }
    }
}

bool GDBSCAN::ExpandCluster(int spIdx, int clusterId) {
//    if (GetDepthPercent({ spIdx }) <= 0)
//    {
//        ChangeClusterId({ spIdx }, UNCLASSIFIED_ID);
//        return false;
//    }
    std::vector<int> seeds = Find_RGBD_Neighbors(spIdx);
    double depPercent = GetDepthPercent(seeds);
    if(depPercent < _minDepPercent)
    {
        //ChangeClusterId(seeds,NOISE_ID);
        return false;
    }
    ChangeClusterId(seeds,clusterId);
    seeds.erase(std::remove(seeds.begin(), seeds.end(), spIdx),seeds.end());
    while(!seeds.empty())
    {
        int currentPt = seeds[0];
        vector<int> result = Find_RGBD_Neighbors(currentPt);
        if(GetDepthPercent(result) >= _minDepPercent) {
            for (auto iter = result.begin(); iter != result.end(); iter++) {
//                if(_SPs[*iter].refinedLabel >= 0 || GetDepthPercent({*iter}) <= 0)
                if(_SPs[*iter].refinedLabel >= 0 )
                    continue;
                if(_SPs[*iter].refinedLabel == UNCLASSIFIED_ID )
                {
                    seeds.push_back(*iter);
                }
                ChangeClusterId({*iter},clusterId);
            }
        }
        seeds.erase(remove(seeds.begin(),seeds.end(),currentPt));
    }
    return true;
}

void GDBSCAN::CLUSTER() {
    int cluster_id = 0;
    for (int i = 0; i < _SPs.size();i++)
    {
        if(_SPs[i].refinedLabel == UNCLASSIFIED_ID)
        {
            if(ExpandCluster(i,cluster_id))
                cluster_id ++;
        }
    }
    //针对缺乏深度信息的，使用rgb距离空间进行聚类

    for (int i = 0; i < _SPs.size();i++)
    {
        if(_SPs[i].refinedLabel == UNCLASSIFIED_ID)
        {
            ExpandClusterUsingRgbDis(i);
        }
    }
}

std::vector<SUPER_PIXEL> GDBSCAN::GetSps(){
    return _SPs;
}
cv::Mat GDBSCAN::GetSpMap(){
    return _slicResult.clone();
}
