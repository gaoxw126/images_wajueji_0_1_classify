https://gitbook.cn/books/5b69828cc129dc4573de8fa0/index.html

尽管没有 GPU，但还是很想在个人笔记本的 CPU 上试试图片分类，有木有？如果你想试试，那就来吧。我们都知道，深度学习在图片分类方面已经取得了很大的成就，大家对于常见的神经网络，如 CNN，VGG 等也非常熟悉，通过 Mnist 或者猫狗大战等数据集都有练习。但是如果你想用自己手里的图片数据集或者还没有数据集，如何从 0 - 1 完成一个图片分类器训练呢？下面笔者将带你完成整个过程。

本场 Chat 你将学到：

自己动手写爬虫爬取图片；
图片预处理，包括 Reshape、灰度、像素归一化等；
用传统机器学习训练图片分类器；
用神经网络训练图片分类器；
模型部署，如何把模型发布成服务。
本次使用的开发环境：

Python3.6
Jupyter Notebook
Keras
下面开始今天的内容。

自己动手写爬虫爬取图片
本次，我们主要尝试对爬来的挖掘机种类进行图像分类。

enter image description here

整个爬取过程中，重点步骤：

    #定义获取总页数的页面
    def get_allpages(res,pattern):
        all_pages = res.find_all('div', 'digg')
        all_page = all_pages[0].find_all('span')[-1].text
        match = pattern.search(all_page).group()
        print("总页数长度" + str(match))
        return int(match)
获取图片 url：

    #获取所有图片的url
    def get_images(res_pic,imgre):
        all_pictures = res_pic.find_all('div', 'searchResultBox')
        pictures = [ x.find_all('img')[0]   for x in all_pictures]
        images = [ imgre.search(str(x)).group().strip()[5:-1] for x in pictures]
        return images
最后进行图片保存：

    def save_images(root,target,images_list):
        path = root + target.split("/")[-1]
        print(path)
        try:
            if not os.path.exists(path):
                os.mkdir(path)
            for x in range(len(images_list)):
                r = requests.get(images_list[x])
                r.raise_for_status()
                #使用with语句可以不用自己手动关闭已经打开的文件流
                file = path + "//" + str(x) + ".png"
                print(file)
                with open(file,"wb") as f: #开始写文件，wb代表写二进制文件
                    f.write(r.content)
                print("文件保存完成")     
        except Exception as e:
            print("文件保存失败:"+str(e))
完整的代码，这里不再赘述，最后我会给出完整代码下载地址。

图片预处理，包括 Reshape、灰度、像素归一化等
下面进行图片的一下预处理，通过 Image 库来处理。

第一步，打开图片，重新定义图片大小并进行图片灰度。

    #定义图像读取
    def  get_image_pixel(file):
        img = Image.open(file)
        img = img.resize((WIDTH,HEIGHT))
        #图片灰度化
        img = img.convert("L")
        img_array = img_to_array(img)
        return img_array
第二步，图片的归一化，图片归一化很简单，因为像素值大小是 0 - 255 之间，所以只要让像素值除以 255，就可以得到 0 - 1 的值，进行了归一化操作。

    X_train = X_train.astype('float32')     
    X_train /= 255   #归一化
    Y_train = np.array(Y)   
用传统机器学习训练图片分类器
这里主要是想说明，我们可以使用非神经网络的有监督或者无监督学习算法来进行图片分类操作。

比如：有监督学习算法：决策树、KNN、SVM、集学习中的随机森林、Xgboost 等；无监督学习的 K - means 等其他聚类算法。

如果使用决策树模型：

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y_train)
    print(clf.score(X_test, y_test))
或者使用 Xgboost 模型：

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.fit(X, y_train)
    print(clf.score(X_test, y_test))
用神经网络 CNN 训练图片分类器
神经网络的图片分类算法有很多，比如 CNN，ResNet50，VGG16 等。下面给出 Keras 简单的 CNN 序贯模型的实现。

    # 构建一个较为简单的CNN模型
    model = Sequential()
    model.add(Convolution2D(32, (2, 2), input_shape=(WIDTH, HEIGHT, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (2, 2)))  
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))                
    model.add(Flatten()) 
    model.add(Dense(hidden_neurons))
    model.add(Activation('relu'))      
    model.add(Dense(classes))
    model.add(Activation('softmax'))
在使用 Keras 时，需要注意一下几点：

1.类别编码
    Y_train = np_utils.to_categorical(Y_train, classes)   #类别编码
2.模型的输入 shape
    total_input = len(X)
    X_train = np.array(X)
    X_train = X_train.reshape(total_input, WIDTH, HEIGHT, 1)
    X_train = X_train.astype('float32')    
3.多分类，一定是 softmax
     model.add(Activation('softmax')
模型部署，如何把模型发布成服务
我们不推荐使用 pickle 或 cPickle 来保存 Keras 模型。

你可以使用 model.save(filepath) 将 Keras 模型和权重保存在一个 HDF5 文件中，该文件将包含：

模型的结构，以便重构该模型；
模型的权重；
训练配置（损失函数，优化器等）；
优化器的状态，以便于从上次训练中断的地方开始。
使用 keras.models.load_model(filepath) 来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译。

    from keras.models import load_model

    model.save('my_model.h5')  
    del model 
    model = load_model('my_model.h5')
下面我们来看看，Python 中如何把模型发布成一个微服务呢？

这里给出 2 个微服务框架 Bottle 和 Flask ，详情可以参考说明文档。

1.安装
Bottle 和 Flask 的安装，分别执行如下命令即可安装成功：

    pip install bottle
    pip install Flask
安装好之后，引入需要的包就可以写微服务程序了。由于这 2 个框架在使用时，用法、语法结构都差不多，网上 Flask 的中文资料相对多一些，所以这里用 Flask 来举例。

2.第一个最小的 Flask 应用
第一个最小的 Flask 应用看起来会是这样:

    from flask import Flask
    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return 'Hello World!'

    if __name__ == '__main__':
        app.run()
把它保存为 hello.py （或是类似的），然后用 Python 解释器来运行：

    python hello.py
或者直接在 Jupyter notebook 里面执行，都没有问题。服务启动将在控制台打印如下消息：

    Running on http://127.0.0.1:5000/
意思就是，可以通过 localhost 和 5000 端口，在浏览器访问：

enter image description here

这时我们就得到了，服务在浏览器上的返回结果，于是也构建与浏览器交互的服务。

如果要修改服务对于的 ip 地址和端口怎么办呢？只需要修改这行代码，如下修改 ip 地址和端口：

    app.run(host='192.168.31.19',port=8088)
3.Flask 发布一个预测模型
如果你现在有个需求要求你训练的模型和浏览器进行交互，那 Flask 就可以实现。在第一个最小的 Flask 应用基础上，我们增加模型预测接口，这里注意：启动之前把 ip 地址修改为自己本机的地址或者服务器工作站所在的ip地址。

完整的代码如下，首先在启动之前就把模型预加载，加载到内存中，然后重新定义 predict 函数，接受一个参数：

    from sklearn.externals import joblib
    from flask import Flask,request
    app = Flask(__name__)

    #获取要预测的图片的像素矩阵
    def get_test_image_pixel(file):
        X_test = []
        img_array = get_image_pixel(file)
        X_test.append(img_array)
        X_test = np.array(X_test)
        X_test = X_test.reshape(1, WIDTH, HEIGHT, 1)
        X_test = X_test.astype('float32')     
        X_test /= 255   #归一化
        return X_test

    #获取最大预测结果的下标
    def get_max_index(y):
        y_list = y.tolist()[0]
        index = y_list.index(max(y_list))
        return index

    def get_label(index):
        return labels[index]

    @app.route('/')
    def hello_world():
        return 'Hello World!'

    @app.route('/predict/<image>')
    def predict(image):
        image = 'D://wajueji//case//50.png'
        index = get_max_index(model.predict(get_test_image_pixel(image)))
        return get_label(index)

    if __name__ == '__main__':
        #根据base_dirs定义类别列表
        #base_dirs = ['caterpillar','case','kubota','doosan','komatsu','sany','volvo','xcmg']
        labels = ['卡特彼勒','凯斯','久保田','斗山','小松','三一重工','沃尔沃','徐工']
        model = load_model('my_model.h5')
        app.run(host='192.168.31.19')
访问就可以得到结果：

enter image description here

总结
本篇通过自己动手写爬虫爬取图片，并进行图片预处理，包括 Reshape、灰度、像素归一化等，用神经网络训练图片分类器，最后讲述了模型部署，介绍了 Python 的 2 个轻量级微服务框架 Bottle 和 Flask，通过 Flask 制作了一个简单的微服务预测接口，实现模型的预测和浏览器交互功能。但是问题远远没有结束，以上只是挖掘机的几种，如果把市场几十种挖掘机和不同型号的挖掘机图片分类，事情可能变得不一样了，这个后续希望大家继续思考。

本文涉及源码下载地址：github

