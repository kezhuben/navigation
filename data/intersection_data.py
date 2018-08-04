#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/11/14 15:26
# desc: get the intersection data from osm.
# ======================================================

from functools import reduce
from urllib import request, error
from utils.gps import get_distance_hav
from utils.decode_osm_lnglat import decode
import numpy as np
import json
import time
import traceback
import sys

sys.path.append('../../Navigation')


def format_intersection_data(roads, times):
    """format the data into proper type, so that it can be plotted more better.
       if the navigation road is much more longer than the main road,
         then need to adjust its size, the opposite is similar.

    Args:
        roads: `list`, [{isNavigationRoad:, lat:[...], lng:[...], type:}, {...}, ...]
        times: `float`, a multiple of the minimum length.

    Returns:
        formatted_result: `list`, [{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...]
    """

    def compute_distance(road):
        """compute the total length of the road."""
        total_size = 0
        length = len(road['lat'])
        for i in range(length - 1):
            total_size += get_distance_hav(road['lat'][i], road['lng'][i], road['lat'][i + 1], road['lng'][i + 1])
        return total_size

    distance_list = list(map(lambda x: compute_distance(x), roads))

    min_distance = min(distance_list)
    min_dist_index = distance_list.index(min_distance)

    # e.g.: [{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...]
    formatted_result = [roads[min_dist_index]]
    roads.pop(min_dist_index)

    def interpolate(start_lng, start_lat, end_lng, end_lat, interp_nums):
        """linear interpolate.
        Args:
            interp_nums: `int`, the number of segments.

        Returns:
            the first interpolated lng/lat: `object`, {"lng":,"lat":}
        """
        lng_val = np.linspace(start_lng, end_lng, interp_nums + 1)
        lat_interp = np.interp(lng_val, [start_lng, end_lng], [start_lat, end_lat])
        interp_result = {"lng": lng_val[1], "lat": lat_interp[1]}

        return interp_result

    # convert all the roads size into a proper length.
    for i in range(len(roads)):
        old_road = roads[i]
        new_road = {
            'isNavigationRoad': old_road['isNavigationRoad'],
            'type': old_road['type']
        }
        road_size = len(old_road['lat'])
        new_road_lng = []
        new_road_lat = []
        road_distance = 0

        new_road_lng.append(old_road['lng'][0])
        new_road_lat.append(old_road['lat'][0])

        ## TODO: delete
        print("min_distance " + str(min_distance))
        print("old_road " + str(compute_distance(old_road)))

        for k in range(road_size - 1):
            segment_distance = get_distance_hav(old_road['lat'][k], old_road['lng'][k], old_road['lat'][k + 1],
                                                old_road['lng'][k + 1])
            ## TODO: delete
            print("segment_distance " + str(segment_distance))

            if road_distance + segment_distance > times * min_distance:
                # over_times = segment_distance // (road_distance + segment_distance - times * min_distance)
                over_times = segment_distance // (times * min_distance - road_distance)
                interp_result = interpolate(old_road['lng'][k], old_road['lat'][k], old_road['lng'][k + 1],
                                            old_road['lat'][k + 1],
                                            over_times + 1)

                # TODO: delete
                print("over_times " + str(over_times))
                print("segment_distance lng/lat " + str(interp_result))
                print("new segment_distance " + str(
                    get_distance_hav(old_road['lat'][k], old_road['lng'][k], interp_result['lat'],
                                     interp_result['lng']
                                     )))

                new_road_lng.append(interp_result['lng'])
                new_road_lat.append(interp_result['lat'])
                break
            else:
                new_road_lng.append(old_road['lng'][k + 1])
                new_road_lat.append(old_road['lat'][k + 1])
                road_distance += segment_distance
        ## TODO: delete
        print("road_distance " + str(road_distance))
        print("-----------------------------------")

        new_road['lng'] = new_road_lng
        new_road['lat'] = new_road_lat
        formatted_result.append(new_road)

    return formatted_result


def get_intersection_data(start_lng, start_lat, end_lng, end_lat):
    """get the intersection data.

    Returns:
        roads_data: [[{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...], [...], ].
    """

    crawl_url0 = "http://121.46.20.210:8007/route?api_key=valhalla-7UikjOk&json={%22locations%22:[{%22lat%22:"
    crawl_url1 = ",%22lon%22:"
    crawl_url2 = "},{%22lat%22:"
    crawl_url3 = ",%22lon%22:"
    crawl_url4 = "}],%22costing%22:%22auto%22,%22directions_options%22:{%22units%22:%22km%22,%22language%22:%22zh" \
                 "-CN%22,%22user_intersection_shap%22:%22true%22}}"

    # joint the url.
    crawl_url = crawl_url0 + str(start_lat) + crawl_url1 + str(start_lng) + crawl_url2 + str(
        end_lat) + crawl_url3 + str(end_lng) + crawl_url4
    html = []
    try:
        response = request.urlopen(crawl_url)
        html = response.read()
    except error:
        print(error.HTTPError.code)
        print(error.HTTPError.msg)

    hjson = json.loads(html.decode('utf-8'))
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " " + str(hjson['trip']['status']) + "\r\n")

    # e.g.: [[{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...], [...], ].
    total_result = []
    # intersections type.
    # intersections_type = []

    if hjson['trip']['status'] == 0:
        shape = hjson['trip']['legs'][0]['shape']
        # e.g.: {"lng":[...],"lat":[...]}
        shape_list = decode(shape)
        maneuvers = hjson['trip']['legs'][0]['maneuvers']

        maneuvers_len = len(maneuvers)
        for i in range(1, maneuvers_len):

            # e.g.: [{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...]
            one_intersection_data = []

            pre_maneuver = maneuvers[i - 1]
            maneuver = maneuvers[i]
            intersections = maneuver['intersections']
            begin_shape_index = maneuver['begin_shape_index']
            end_shape_index = maneuver['end_shape_index']
            pre_begin_shape_index = pre_maneuver['begin_shape_index']
            pre_end_shape_index = pre_maneuver['end_shape_index']

            if len(intersections) >= 1:

                # previous lng/lat.
                pre_index_list = list(range(pre_begin_shape_index, pre_end_shape_index + 1))
                # pre_lng = list(map(lambda x: shape_list["lng"][x], pre_index_list))
                # pre_lat = list(map(lambda x: shape_list["lat"][x], pre_index_list))
                pre_lng = []
                pre_lat = []

                # intersection lng/lat.
                for intersection in intersections:
                    intersection_lng_lat = decode(intersection)
                    intersection_lng_lat["lng"] = pre_lng + intersection_lng_lat["lng"]
                    intersection_lng_lat["lat"] = pre_lat + intersection_lng_lat["lat"]
                    intersection_lng_lat["isNavigationRoad"] = False
                    intersection_lng_lat["type"] = maneuver['type']
                    one_intersection_data.append(intersection_lng_lat)

                # main road lng/lat.
                main_index_list = list(range(begin_shape_index, end_shape_index + 1))
                main_lng = list(map(lambda x: shape_list["lng"][x], main_index_list))
                main_lat = list(map(lambda x: shape_list["lat"][x], main_index_list))
                main_road = {
                    "isNavigationRoad": True,
                    "lng": pre_lng + main_lng,
                    "lat": pre_lat + main_lat,
                    "type": maneuver['type']
                }
                one_intersection_data.append(main_road)
                total_result.append(one_intersection_data)

    # filter the data, delete the road whose size < 2.
    def filter_roads(road, size):
        bool_list = list(map(lambda x: len(x['lng']) > size, road))
        bool_final = reduce(lambda x, y: x & y, bool_list)
        return bool_final

    filter_total_result = list(filter(lambda x: filter_roads(x, 1), total_result))

    formatted_result = list(map(lambda x: format_intersection_data(x, 3), filter_total_result))
    print(formatted_result)

    # TODO: delete
    def compute_distance(road):
        """compute the total length of the road."""
        total_size = 0
        length = len(road['lat'])
        for i in range(length - 1):
            total_size += get_distance_hav(road['lat'][i], road['lng'][i], road['lat'][i + 1], road['lng'][i + 1])
        return total_size

    distance_list = list(map(lambda x: list(map(lambda y: compute_distance(y), x)), formatted_result))
    print(distance_list)

    return formatted_result


if __name__ == '__main__':
    roads = get_intersection_data(122.92118511199953, 31.236532062402154, 111.58457832336427, 23.11589095262163)
    # print(roads)
    from plot.plot_crossroad import get_crossroad_image
    for i in range(len(roads)):
        get_crossroad_image(roads[i], save_path="C:\\Users\\yangcheng\\Desktop\\交叉路识别\\数据")
