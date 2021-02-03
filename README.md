这是一个侦测倾斜文本的检测网络，网络修改自yolov4。网络通过学习四个点在非旋转矩形上的偏移来定位出一个四边形来表示一个物体。

![image-20210203091321721](images\image-20210203091321721.png)



# 要求

Python 3.8 或更晚，[安装.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)依赖项，包括 。要安装运行：`torch>=1.7`

```
$ pip install -r requirements.txt
```



# 教程

训练自定义数据：

图片路径放置于：MONet\data\image下

文本标签，放置于：MONet\data下

```
2007_000042.jpg  train  118.1955  166.6799  236.391  272.0  0.0  train  380.6955  164.6102  236.609  265.8605  0.0
2007_000027.jpg  person  251.1765  217.2762  88.1079  269.8248  2.641593
2007_000039.jpg  moniter  261.6014  181.5507  207.3795  185.8652  0.01
2007_004231.jpg  shap  265.5229  125.643  420.7225  59.8207  0.49
2007_002094.jpg  bird  212.9723  166.3789  129.4122  312.5744  0.6
2007_000733.jpg  person  186.7624  204.1546  149.8812  379.3314  0.4
2007_004841.jpg  plane  141.0372  226.6162  69.8926  44.4233  2.531593  plane  361.3566  117.5155  25.3169  21.1437  2.461593
```

文本格式上图所示，

每一排是一张照片的数据，第一项是图片名称，之后是类别名，（c_x,c_y,w,h,angle）依次重复目标标签信息。

可多显卡训练。

