# C++ 实现简单的线性回归

本文将用C++实现一个基于梯度下降的一元线性回归模型。
首先在头文件中定义LinearRegression类中包含的参数和方法：

```C++
#ifndef ML_LINEARREGRESSION_H//如果这个宏没有被定义
#define ML_LINEARREGRESSION_H//则定义宏


//代码参考https://github.com/aluxian/CPP-ML-LinearRegression/blob/master/LinearRegression.h

class LinearRegression {
    public:
        //特征，使用指针是因为特征是数组形式
        double *x;
        //预测值
        double *y;
        //样本数量
        int m;
        //系数
        double *theta;
        //创建实例
        LinearRegression(double x[],double y[],int m);
        //训练
        void train(double alpha,int iterations);
        //预测
        void predict(double x);
    private:
        //计算模型损失
        static double compute_cost(double x[],double y[],double theta[],int m);
        //计算单个预测值
        static double h(double x,double theta);
        //预测
        static double *calculate_predictions(double x[],double theta[],int m); 
        //梯度下降
        static double *gradient_descent(double x[],double y[],double alpha,int iter,double *J,int m);
};

#endif
```

定义好LinearRegression类之后，在LinearRegression.cpp文件中首先要对类进行初始化：
```C++
#include <iostream>
#include "linear_regression.h"
#include "Utils.h"

using namespace std;

//初始化
LinearRegression::LinearRegression(double x[],double y[],int m) {
    this->x = x;
    this->y = y;
    this->m = m;
}
```

接下来定义训练模型所需要的函数：
## 1. 梯度下降函数

```C++
//梯度下降
double *LinearRegression::gradient_descent(double x[],double y[],double alpha;int iters;double *J;int m) {
    double *theta = new double[2];
    theta[0] = 1;
    theta[1] = 1;
    for (int i=0;i<iters;i++) {
        double *predictions = calculate_predictions(x,theta,m);
        double *diff = Utils::array_diff(predictions,y,m);
        double *error_x1 = diff;
        double *error_x2 = Utils::array_multiplication(diff,x,m);
        theta[0] = theta[0] - alpha*(1.0/m) * Utils::array_sum(error_x1,m);
        theta[1] = theta[1] - alpha*(1.0/m) * Utils::array_sum(error_x2,m);
        J[i] = compute_cost(x,y,theta,m);
    }
    return theta;
}
```

## 2. 训练函数

```C++
void LinearRegression::train(double alpha,int iterations) {
    double *J = new double[iterations];//将J定义为一个包含了iterations个元素的数组
    this->theta = gradient_descent(x,y,alpha,iterations,J,m);
    cout << "J = ";
    for (int i =0; i< iterations;++i) {
        cout << J[i] << " ";
    }
    cout << endl << "Theta: " << theta[0] << " " << theta[1] << endl;
}
```
## 3. 预测与误差函数

```C++

//预测
double LinearRegression::predict(double x) {
    return h(x,theta);
}

//计算误差
double LinearRegression::compute_cost(double x[],double y[],double theta[],int m) {
    double *predictions = calculate_predictions(x,theta,m);
    double *diff = Utils::array_diff(predictions,y,m);
    double *sq_errors = Utils::array_pow(diff,m,2);
    return (1.0/(2*m)) * Utils::array_sum(sq_errors,m);
}

//预测单个值
double LinearRegression::h(double x,double theta[]) {
    return theta[0] + theta[1] * x;
}

//预测
double *LinearRegression::calculate_predictions(double x[],double theta[],int m) {
    double * predictions = new double[m];
    for (int i=0;i<m;i++) {
        predictions[i] = h(x[i],theta);
    }
    return predictions;
}
```
