def pointInPolygon(point,polygon):
    j = len(polygon) - 1
    result = False
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[j]
        #如果point正好是polygon上的顶点
        if (p1[0] == point[0] and p1[1] == point[1]) or (p2[0] == point[0] and p2[1] == point[1]):
            return True
        j = i
        if point[1] > min(p1[1],p2[1]):
            if point[1] <= max(p1[1],p2[1]):#如果point的纵坐标在线段p1p2的范围内
                if point[0] < max(p1[0],p2[0]):
                    if p1[1] != p2[1]:#如果线段p1p2不是水平线段，则求过point的水平直线与线段p1p2的交点
                        x = (point[1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
                    if p1[0] == p2[0] or point[0] < x:
                        result = not result
                    if point[0] == x:
                        return True
    return result

if __name__ == "__main__":
    p = (1,1)
    polygon = [(2,1),(3,2),(10,0),(2,3),(1,2)]
    print(pointInPolygon(p,polygon))
    p2 = (0,2)
    print(pointInPolygon(p2,polygon))