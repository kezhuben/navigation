#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/11/10 10:57
# desc: generate the images data for CNNs.
# ======================================================

import traceback
import requests
from data.points_data import get_user_stay_points_from_db
from plot.plot_crossroad import get_crossroad_image
from data.intersection_data import get_intersection_data
from db.insert import batch_insert
import sys

sys.path.append('../../Navigation')

TEMP_SAVE_PATH = "C:\\Users\\yangcheng\\Desktop\\交叉路识别\\数据"
SAVE_PATH = '/data/bigData/images_data/crossroad_images'
LOG_PATH = './log/images_generation_log_file.log'
DB_CONF = {
    'database': "bigData",
    'user': "postgres",
    'password': "zzc709394",
    'host': "121.46.20.195",
    'port': "5432"
}


def generate_intersection_images():
    """generate intersection images and then store in '195' postgresql."""

    # log file.
    log = open(LOG_PATH, 'a')

    # e.g.: [{start_lng:,start_lat:,end_lng:,end_lat:,}, {...}, ...]
    start_end_points = get_user_stay_points_from_db()

    # postgres table schema.
    pg_table_schema = {
        'table_name': "crossroad_images",
        'columns_name': ["path", "label", "label_id", "road_data"]
    }

    # within every navigation:
    for navigation in range(len(start_end_points)):
        # e.g.: [[{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...], [...], ]
        intersection_data = get_intersection_data(navigation['start_lng'], navigation['start_lat'],
                                                  navigation['end_lng'], navigation['end_lat'])
        # generate the images for every intersection.
        batch_intersection_data = []
        for i in range(len(intersection_data)):
            i_image_path = get_crossroad_image(intersection_data[i], save_path=TEMP_SAVE_PATH)
            batch_intersection_data.append(
                [i_image_path, "null", intersection_data[0]['type'], str(intersection_data[i])])
        # insert into postgres.
        batch_insert(DB_CONF, pg_table_schema, batch_intersection_data)


if __name__ == "__main__":
    pass
