//
// Created by laiwenjie on 2021/10/28.
//

#ifndef DEPTHINPAINTER_MAT_VECTOR_H
#define DEPTHINPAINTER_MAT_VECTOR_H
#include <opencv2/opencv.hpp>
class mat_vector:public std::vector<cv::Mat>{
//private:
//    int _size = 0;
public:
    void addItem(cv::Mat item){
        this->push_back(item);
        width = item.cols;
        height = item.rows;
        dtype=item.type();
    }
    mat_vector operator+(const mat_vector&b){
        mat_vector vec;
        assert(this->size() == b.size());
        for(int i = 0;i<this->size();i++)
        {
            cv::Mat sum;
            sum = this->operator[](i) +  b[i];
            vec.push_back(sum);
        }
        return vec;
    }
    mat_vector operator-(const mat_vector&b) {
        mat_vector vec;
        assert(this->size() == b.size());
        for(int i = 0;i<this->size();i++)
        {
            cv::Mat sum;
            sum = this->operator[](i) -  b[i];
            vec.push_back(sum);
        }
        return vec;
    }

    template<typename T>
    mat_vector operator*(T b)
    {
        mat_vector vec;
        for(int i = 0;i<this->size();i++)
        {
            cv::Mat tmp;
            tmp = this->operator[](i)*b;
            vec.push_back(tmp);
        }
        return vec;
    }
    int width = 0, height = 0, dtype=0;
};
#endif //DEPTHINPAINTER_MAT_VECTOR_H
