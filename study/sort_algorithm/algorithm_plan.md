# 算法学习计划

时间：2018年3月27日14:16:19

## 1. 各种排序算法原理以及效率对比


## 2. 动态规划算法举例——加权有向图中倒最短路径算法

使用邻接矩阵倒方式来表示这个图，定义边的数据结构

```python

class DirectedEdge():

    self.weight = 0
    self.last = ""
    self.next = ""

    def __init__(weight,last,next):
        self.weight = weight
        self.last = last
        self.next = next
    
```

定义邻接表矩阵

```python

class EdgeWeightDiGraph():

    self.V = 0#点的数量
    self.E = 0#边的数量
    self.edges = []

    def __init__(edges,V,E):
        self.edges = edges
        self.V = V
        self.E = E 
```




