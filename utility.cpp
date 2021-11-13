//
// Created by bytelai on 2021/11/8.
//

#include "utility.h"
using namespace cv;
double MinNoZero(Mat input, Mat msk)
{
    double *ptr = (double *) input.data;
    double min = DBL_MAX;
    if(msk.empty()) {
        for (int i = 0; i < input.cols * input.rows; i++) {
            if (*ptr < min && *ptr != 0)
                min = *ptr;
            ptr++;
        }
    }
    else
    {
        uchar *mPtr = (uchar *) msk.data;
        for (int i = 0; i < input.cols * input.rows; i++) {
            if (*ptr < min && *ptr != 0 && *mPtr == 1)
                min = *ptr;
            ptr++;
            mPtr++;
        }
    }
    return min;
}
