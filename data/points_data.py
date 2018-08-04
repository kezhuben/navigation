#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/11/10 12:03
# desc: get the stay points data from database.
# ======================================================

import json
import psycopg2
import pandas as pd


def get_user_stay_points_from_db():
    db = psycopg2.connect(database="db_zuji", user="postgres", password="zzc709394", host="121.46.20.195", port="5432")
    cursor = db.cursor()

    query = """
select c.* from (
  select a.muid,a.status,a.start_time,a.end_time,b.points from block_data_backend_analyse_2017 a INNER JOIN points_data_backend_analyse_2017 b 
    on  a.block_id = b.block_id
    where a.status = 'stay' and a.muid != '-1' and a.muid != '0' and a.start_time != 0
    order by a.start_time
)c
  ORDER BY c.muid,c.start_time"""
    cursor.execute(query)

    # [(muid, status, start_time, end_time, [{lng:, lat:}]), (...), ...]
    fetch_result = cursor.fetchall()
    # parser points dada.
    parser_result = list(map(lambda x: (x[0], x[1], x[2], x[3], json.loads(x[4])[0]['lng'], json.loads(x[4])[0]['lat']),
                             fetch_result))

    # convert into dataframe: [(muid, dataframe), (...), ...]
    df_resutl = pd.DataFrame(parser_result, columns=['muid', 'status', 'start_time', 'end_time', 'lng', 'lat'])
    groups_result = list(df_resutl.groupby('muid'))
    filter_result = list(filter(lambda group: len(group[1]) != 1, groups_result))

    # format the points into a chain list.
    def format_points(dataframe):
        points = []
        for i in range(len(dataframe) - 1):
            points.append({
                'start_lng': dataframe.iat[i, 4],
                'start_lat': dataframe.iat[i, 5],
                'end_lng': dataframe.iat[i + 1, 4],
                'end_lat': dataframe.iat[i + 1, 5],
            })
        return points

    # [ [{start_lng:,start_lat:,end_lng:,end_lat:,}, {start_lng:,start_lat:,end_lng:,end_lat:,}, ...], [{...}, ...], ...] ]
    format_result = list(map(lambda x: format_points(x[1]), filter_result))
    # flatten, convert into [{start_lng:,start_lat:,end_lng:,end_lat:,}, {...}, ...]
    output_result = sum(format_result,[])
    return output_result


# print(len(get_user_stay_points_from_db()))
# print(get_user_stay_points_from_db())
