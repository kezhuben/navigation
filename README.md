# 探图离线导航


## 项目结构

---
```python
├── data
│   ├── images_data.py
│   ├── __init__.py
│   ├── intersection_data.py
│   ├── MNIST_data
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   ├── osm_shape.py
│   ├── points_data.py
│   ├── post_turnbyturn2ai.pycd 
│   ├── roads_data.py
│   └── roads_data.txt
├── db
│   ├── __init__.py
│   └── insert.py
├── generate_images.py
├── __init__.py
├── log
│   ├── images_generation_log_file.log
│   ├── images_recognization_log_file.log
│   └── __init__.py
├── models
│   ├── dense_nets_model
│   │   ├── checkpoint
│   │   ├── dense_nets.ckpt.data-00000-of-00001
│   │   ├── dense_nets.ckpt.index
│   │   ├── dense_nets.ckpt.meta
│   │   ├── __init__.py
│   ├── __init__.py
├── nets
│   ├── dense_net3.py
│   ├── dense_nets2.py
│   ├── dense_nets.py
│   ├── faster_rcnn.py
│   ├── __init__.py
├── nets_params
│   ├── dense_nets_params.py
│   ├── faster_rcc_params.py
│   ├── __init__.py
├── plot
│   ├── __init__.py
│   ├── plot_crossroad.py
├── README.md
├── server.py
├── test.py
└── utils
    ├── decode_osm_lnglat.py
    ├── gps.py
    ├── __init__.py
```

- data: 数据目录，包含一些测试用例数据；

  - intersection_data：接口获取交叉路数据；
  - points_data：利用足迹用户轨迹数据批量生成导航的起点终点经纬度；
  - roads_data：获取足迹用户轨迹数据中的运动块数据（针对新西兰）；
- models: 算法模型目录，包含算法训练结果保存的模型，目前保存的为dense-net在mnist数据上的模型；
  
  - meta 文件保存了graph结构,包括 GraphDef, SaverDef等,当存在meta时,我们可以不在文件中定义模型,也可以运行,而如果没有meta时,我们需要定义好模型,再加载data file,得到变量值.
  - index 文件为一个 string-string table,table的key值为tensor,value为BundleEntryProto, BundleEntryProto.
  - data 文件保存了模型的所有变量的值.
  
- nets: 算法目录，包含目前比较前沿的一些CNN算法，目前主要有dense-net和faster-rcnn；
- nets_params: 算法参数，对应nets目录下的算法；
- plot: 图形生成，目前主要针对交叉路的图像生成；
- db：数据库的一些操作；
- log：log文件；
- utils: 一些工具类；

server.py 是plot的接口，对接osm数据生成图片，打标签，保存入库。
generate_images.py 是生成交叉路图片，并保存在服务器和数据库的脚本。

## 环境配置

---
### centos7.X

bazel: https://github.com/bazelbuild/bazel/releases:
```python
unzip bazel-0.5.4-dist.zip
cd bazel-0.5.4
./compile.sh

vim /etc/profile
export PATH=$PATH:/usr/local/bazel-0.5.4/output
```

tensorflow:
```python
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure # 此处根据实际需要进行配置

bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2  //tensorflow/tools/pip_package:build_pip_package # 此处参数根据上一步configure设置

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

cd /tmp/tensorflow_pkg #进入该目录后查看对应的whl版本

pip install tensorflow-1.4.0.dev0-cp27-cp27mu-linux_x86_64.whl
```

virtualenv for python3:
```python
apt-get install virtualenvwrapper
pip install virtualenvwrapper

# 配置 ~/.bashrc
vim ~/.bashrc
soucre /usr/share/virtualenvwrapper/virtualenvwrapper.sh

# 创建python3虚拟环境
mkvirtualenv -p /usr/bin/python3.5 env35

# 进入 env34
workon env35
# 退出
deactivate
```

其他：
```python
pip install flask
pip install flask_restful
pip install PIL
pip install matplotlib
pip install scipy
pip install numpy
pip install skimage
pip install psycopg2
```

测试：
```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello)) #Hello, TensorFlow!
```


## 交叉路识别

---

### 交叉路例子

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/1.png" width="167" height="167" />
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/2.png" width="167" height="167" />
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/3.png" width="167" height="167" />
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/4.png" width="167" height="167" />
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/5.png" width="167" height="167" />
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/6.png" width="167" height="167" />
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/CNN/7.png" width="167" height="167" />

### 交叉路经纬度数据接口：

```python
http://121.46.20.210:8007/route?api_key=valhalla-7UikjOk&json={%22locations%22:[{%22lat%22:23.136532062402154,%22lon%22:113.32118511199953},{%22lat%22:23.11589095262163,%22lon%22:113.28457832336427}],%22costing%22:%22auto%22,%22directions_options%22:{%22units%22:%22km%22,%22language%22:%22zh-CN%22,%22user_intersection_shap%22:%22true%22}
```
传入起点终点经纬度，直接get即可。

### 交叉路图片数据：


### 路口识别接口：
```python

```

