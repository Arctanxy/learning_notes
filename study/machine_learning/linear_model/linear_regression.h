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