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
    mat_vector(int i):std::vector<cv::Mat>(i){
    }

    mat_vector(int i,cv::Mat element ){
        for(int j = 0; j <i;j++)
            this->addItem(element.clone());
    }

    mat_vector():std::vector<cv::Mat>(){
    }

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
            vec.addItem(sum);
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
            vec.addItem(sum);
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
            vec.addItem(tmp);
        }
        return vec;
    }

    mat_vector mul(mat_vector b)
    {
        assert(this->size() == b.size());
        mat_vector ret;
        for(int i = 0;i<this->size();i++)
        {
            cv::Mat tmp = this->operator[](i).mul(b[i]);
            ret.addItem(tmp);
        }
        return ret;
    }

    mat_vector divide(mat_vector b)
    {
        assert(this->size() == b.size());
        mat_vector ret;
        for(int i = 0;i<this->size();i++)
        {
            cv::Mat tmp;
            cv::divide(this->operator[](i),b[i],tmp);
            ret.addItem(tmp);
        }
        return ret;
    }

    mat_vector clone()
    {
        mat_vector vec;
        for(int i = 0;i<this->size();i++)
        {
            cv::Mat tmp;
            tmp = this->operator[](i).clone();
            vec.addItem(tmp);
        }
        return vec;
    }

    double norm1()
    {
        cv::Mat tmp = cv::Mat::zeros(this->operator[](0).rows,this->operator[](0).cols, this->operator[](0).type());
        for(auto iter = this->begin(); iter!=this->end();iter++)
        {
            tmp += cv::abs(*iter);
        }
        double ret = cv::sum(tmp)[0];
        return ret;
    }
    double norm2()
    {
        cv::Mat tmp = cv::Mat::zeros(this->operator[](0).rows,this->operator[](0).cols, this->operator[](0).type());
        for(auto iter = this->begin(); iter!=this->end();iter++)
        {
            tmp += iter->mul(*iter);
        }
        cv::sqrt(tmp,tmp);
        double ret = cv::sum(tmp)[0];
        return ret;
    }
    int width = 0, height = 0, dtype=0;
};

template<typename T>
mat_vector operator*(T lem,mat_vector mat){
    return mat*lem;
}
#endif //DEPTHINPAINTER_MAT_VECTOR_H
