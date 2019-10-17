'''
线搜索——进退法

另有：
黄金分割法
使用与在[a,b]区间上的单谷函数求极小值问题
只要求是单谷函数，不要求连续
插值法
'''

import numpy as np 

def f(x):
    return x**2 -  6*x + 9

def line_search(f,init_theta1,alpha,gradient):
    theta1 = init_theta1
    theta2 = theta1 - alpha * gradient
    while True:
        if f(theta1) > f(theta2):
            theta3 = theta2 - alpha * gradient
            if f(theta2) < f(theta3):
                dis = (min(theta1,theta3)+max(theta1,theta3))/2
                best_alpha = (dis - init_theta1)/gradient
                return best_alpha
            elif f(theta2) >= f(theta3):
                #递推
                theta1 = theta2
                theta2 = theta3
                theta3 -= alpha*gradient

        elif f(theta1) < f(theta2):
            dis = (min(theta1,theta2) + max(theta1,theta2))/2
            best_alpha = (dis-init_theta1)/gradient
            return best_alpha

print(line_search(f,0,1,-1))