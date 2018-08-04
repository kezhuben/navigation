#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/11/12 23:21
# desc: some gps utils.
# ======================================================

from math import sin, asin, cos, radians, fabs, sqrt

EARTH_RADIUS = 6378.137  # 地球平均半径，6371km

def get_distance_hav(lat0, lng0, lat1, lng1):
    """ 用haversine公式计算球面两点间的距离。"""

    def hav(theta):
        s = sin(theta / 2)
        return s * s

    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h)) * 1000

    return distance

# print(get_distance_hav(22.599578, 113.973129,22.6986848,114.3311032))