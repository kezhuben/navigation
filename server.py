#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/10/26 17:12
# desc: api server contains plotting, nets algorithm request and so on.
# ======================================================

import json
from flask import Flask, request
# from flask.ext.restful import Resource, reqparse
from flask_restful import reqparse

app = Flask(__name__)


@app.route('/navigation/v1.0/get_crossroad_image', methods=['POST'])
def get_crossroad_image():
    j = request.data

    print(json.loads(j)['data'])
    # d = json.loads(j)
    # print(d)
    return j


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8887, debug=True)
