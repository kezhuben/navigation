#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/10/25 15:03
# desc: plot crossroad using matplotlib.
# ======================================================

from PIL import Image
# import data.images_data
import matplotlib.pyplot as plt
import math
import time
import random


def get_crossroad_image(roads_data,
                        save_path,
                        size=5,
                        dpi=100,
                        background_color='#E6EEF3',
                        roads_color='#303A46',
                        navigation_color='#FDFD40'):
    """plot the crossroad, and save the image.

    Args:
        roads_data: `list`, [{isNavigationRoad:, lat:[...], lng:[...]}, {...}, ...].
        save_path: `str`, images save path.
        size: `Int`, figure size.
        dpi: `Int`, dots per inch for output image.
        background_color: `str`, its rgb is (230, 238, 243).
        roads_color: `str`, its rgb is (48, 58, 70).
        navigation_color: `str`, its rgb is (253, 253, 64).

    Returns:
        the image file.
    """
    # initial the figure size, color, dpi, and so on.
    figure = plt.figure(figsize=(size, size), dpi=dpi, facecolor=background_color)
    # figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # add the sub plot, and set the axis background color.
    p = plt.subplot(111)
    p.set_facecolor(background_color)

    # divide the data into navigation and intersection.
    navigation_road = list(filter(lambda road: road['isNavigationRoad'], roads_data))
    intersection = roads_data

    # temp axes limits [xmin, xmax, ymin, ymax].
    navigation_road_lng = navigation_road[0]['lng']
    navigation_road_lat = navigation_road[0]['lat']
    max_lng = max(navigation_road_lng)
    min_lng = min(navigation_road_lng)
    max_lat = max(navigation_road_lat)
    min_lat = min(navigation_road_lat)

    # calculate the image rotated slope, there are four situation needed considered.
    rotate_slope = 0
    line_direction = 1
    if navigation_road_lng[0] != navigation_road_lng[1]:

        rotate_slope = (navigation_road_lat[1] - navigation_road_lat[0]) / (
            navigation_road_lng[1] - navigation_road_lng[0])

        # first quartile
        if navigation_road_lng[1] > navigation_road_lng[0] and navigation_road_lat[1] > navigation_road_lat[0]:
            line_direction = 1
        # second quartile.
        elif navigation_road_lng[1] < navigation_road_lng[0] and navigation_road_lat[1] > navigation_road_lat[0]:
            line_direction = 2
        # third quartile.
        elif navigation_road_lng[1] < navigation_road_lng[0] and navigation_road_lat[1] < navigation_road_lat[0]:
            line_direction = 3
        # forth quartile.
        elif navigation_road_lng[1] > navigation_road_lng[0] and navigation_road_lat[1] < navigation_road_lat[0]:
            line_direction = 4

    # elif navigation_road_lat[1] > navigation_road_lat[0]:
    #     line_direction = 2
    elif navigation_road_lat[1] < navigation_road_lat[0]:
        rotate_slope = 'this happen because the line direction is vertical downward, so this string will != 0 forever.'
        line_direction = 5

    # plot all road.
    road_nums = len(roads_data)
    for i in range(road_nums):
        road_lng = intersection[i]['lng']
        road_lat = intersection[i]['lat']
        p.plot(road_lng, road_lat, lw=60 // road_nums // 1.5, color=roads_color, zorder=5)

        # update the axes limits.
        if max(road_lng) > max_lng: max_lng = max(road_lng)
        if min(road_lng) < min_lng: min_lng = min(road_lng)
        if max(road_lat) > max_lat: max_lat = max(road_lat)
        if min(road_lat) < min_lat: min_lat = min(road_lat)

    # plot the navigation road.
    if len(navigation_road_lng) <= 3:
        p.plot(navigation_road_lng, navigation_road_lat, lw=60 // (road_nums * 2), color=navigation_color, zorder=8)
    else:
        # update the navigation road start and end position.
        new_start_lng = (navigation_road_lng[0] + navigation_road_lng[1]) / 2
        new_end_lng = (navigation_road_lng[-1] + navigation_road_lng[-2]) / 2
        new_start_lat = (navigation_road_lat[0] + navigation_road_lat[1]) / 2
        new_end_lat = (navigation_road_lat[-1] + navigation_road_lat[-2]) / 2
        navigation_road_lng[0] = new_start_lng
        navigation_road_lng[-1] = new_end_lng
        navigation_road_lat[0] = new_start_lat
        navigation_road_lat[-1] = new_end_lat

        p.plot(navigation_road_lng, navigation_road_lat, lw=60 // (road_nums * 2.5), color=navigation_color, zorder=8)

    # plot the arrow(anticlockwise).
    if navigation_road_lng[-1] == navigation_road_lng[-2]:
        if navigation_road_lat[-1] > navigation_road_lat[-2]:
            arrow_angle = 0
        else:
            arrow_angle = 180
    else:
        yticks_arrow = p.get_yticks()[1] - p.get_yticks()[0]
        xticks_arrow = p.get_xticks()[1] - p.get_xticks()[0]
        arrow_ticks_scale = xticks_arrow / yticks_arrow
        arrow_slope = (navigation_road_lat[-1] - navigation_road_lat[-2]) / (
            navigation_road_lng[-1] - navigation_road_lng[-2]) * arrow_ticks_scale
        if arrow_slope < 0:
            if (navigation_road_lat[-1] - navigation_road_lat[-2]) < 0 and (navigation_road_lng[-1] - navigation_road_lng[-2]) > 0:
                arrow_angle = 270 - math.atan(arrow_slope) * 180 / math.pi
            else:
                arrow_angle = 90 + math.atan(arrow_slope) * 180 / math.pi
        else:
            if (navigation_road_lat[-1] - navigation_road_lat[-2]) >= 0 and (navigation_road_lng[-1] - navigation_road_lng[-2]) > 0:
                arrow_angle = - (90 - math.atan(arrow_slope) * 180 / math.pi)
            else:
                arrow_angle = 90 + math.atan(arrow_slope) * 180 / math.pi
    # print(navigation_road_lat[-1],navigation_road_lat[-2], navigation_road_lng[-1], navigation_road_lng[-2])

    p.plot(navigation_road_lng[-1],
           navigation_road_lat[-1],
           marker=(3, 0, arrow_angle),
           markersize=60 // road_nums // 1.1,
           color=navigation_color,
           zorder=10)

    # fine tuning the coordinate.
    # figure.subplots_adjust(left=0.1,right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    # figure.subplots_adjust(left=min_lng, right=max_lng, bottom=min_lat , top=max_lat, wspace=0, hspace=0)
    # p.axis([min_lng , max_lng , min_lat , max_lat ])
    # p.axis('equal')
    p.axis('off')
    # plt.xticks([])
    # plt.yticks([])

    figure_name = str(int(time.time())) + str(random.randint(0, 9)) + '.png'
    full_path = save_path + '\\' + figure_name
    # save the figure.
    plt.savefig(full_path,
                dpi=dpi,
                facecolor=background_color,
                edgecolor=background_color,
                # orientation=80,
                papertype=None,
                format=None,
                transparent=False,
                bbox_inches=None,
                pad_inches=0.1,
                frameon=None)
    plt.close('all')
    # rotate the image(anticlockwise).
    rotate_angle = 0
    if rotate_slope != 0:
        yticks_interval = p.get_yticks()[1] - p.get_yticks()[0]
        xticks_interval = p.get_xticks()[1] - p.get_xticks()[0]
        ticks_scale = xticks_interval / yticks_interval
        if line_direction == 1:
            rotate_angle = 90 - math.atan(rotate_slope * ticks_scale) * 180 / math.pi
        elif line_direction == 2:
            rotate_angle = -math.atan(rotate_slope * ticks_scale) * 180 / math.pi - 90
        elif line_direction == 3:
            rotate_angle = 270 - math.atan(rotate_slope * ticks_scale) * 180 / math.pi
        elif line_direction == 4:
            rotate_angle = -math.atan(rotate_slope * ticks_scale) * 180 / math.pi + 90
        elif line_direction == 5:
            rotate_angle = 180

        pixel = size * dpi
        img = Image.open(full_path)
        # img.show()
        new_img = Image.new('RGB', (pixel, pixel), background_color)
        # do not tailor the rotated image(expand = True).
        rotate_img = img.rotate(rotate_angle, expand=True).resize((pixel, pixel))

        new_img.paste(rotate_img, (0, 0, pixel, pixel), rotate_img)
        # new_img.show()
        new_img.save(full_path)
        # print(full_path)
        return full_path


# get_crossroad_image(roads_data=data.images_data.data, save_path='C:\\Users\\yangcheng\\Desktop', dpi=250)
