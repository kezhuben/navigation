#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/11/12 10:21
# desc: get the navigation roads data.
# ======================================================

import json
import psycopg2
import pandas as pd
from utils.gps import get_distance_hav
from functools import reduce


def get_user_moving_roads_from_db():
    db = psycopg2.connect(database="db_zuji", user="postgres", password="zzc709394", host="121.46.20.195", port="5432")
    cursor = db.cursor()

    query = """
select g.points from (

select d.* from block_data_backend_analyse_2017 d where d.muid in (

  select c.muid from (
    
    select a.muid,b.* from block_data_backend_analyse_2017 a INNER JOIN poi_data_backend_analyse_2017 b on a.block_id = b.block_id
    where (b.region_cn = '新西兰' or b.region_en = 'New Zealand') and a.muid != '-1'

  )c

)

)f INNER JOIN points_data_backend_analyse_2017 g on f.block_id = g.block_id where f.status = 'moving' """

    cursor.execute(query)

    # [('[{"lng":, "lat": }, {...}, ...]'), (...), ...]
    fetch_result = cursor.fetchall()

    # compute the max distance within every route.
    def compute_max_distance(data):
        # [{"lng":, "lat": }, {...}, ...]
        parser_object = json.loads(data)
        parser_object.pop(0)

        length = len(parser_object)
        distance = 0
        if length <= 2:
            return 100000000
        else:
            for i in range(length - 1):
                lat0 = parser_object[i]['lat']
                lng0 = parser_object[i]['lng']
                lat1 = parser_object[i + 1]['lat']
                lng1 = parser_object[i + 1]['lng']
                new_distance = get_distance_hav(lat0, lng0, lat1, lng1)
                if new_distance > distance:
                    distance = new_distance
            return distance

    # distance list, e.g.: [1000,2000,414,41,41,4141324, ....]
    distances = sorted(list(map(lambda x: compute_max_distance(x[0]), fetch_result)), reverse=True)
    print(distances[1:800])

    # filter the data.
    filter_result = list(filter(lambda x: compute_max_distance(x[0]) < distances[800], fetch_result))
    print(filter_result)

    def parser(data):
        # [{"lng":, "lat": }, {...}, ...]
        parser_object = json.loads(data)
        # parser_object.pop(0)

        def map_to_string(lng_lat_object):
            lng = lng_lat_object["lng"]
            lat = lng_lat_object["lat"]
            return '{},{};'.format(lng, lat)

        # convert into {"ROAD_LINE": 'lng,lat;lng,lat; ...'}
        object_value = map(lambda x: map_to_string(x), parser_object)
        return {"ROAD_LINE": reduce(lambda x, y: x + y, object_value)}

    # parser and format the result, e.g.:  [{"ROAD_LINE": 'lng,lat;lng,lat; ...'}, {...}, ...]
    formatted_result = list(map(lambda x: parser(x[0]), filter_result))

    with open('./roads_data.txt', 'a+', encoding='utf-8') as f:
        for i in range(len(formatted_result)):
            f.write(str(formatted_result[i]) + ',' + '\n')
    f.close()

    return formatted_result


if __name__ == '__main__':
    get_user_moving_roads_from_db()
