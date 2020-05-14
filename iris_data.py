# -*- coding: utf-8 -*-

#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt   #绘图调用的包
import numpy as np  #数学计算，产生标号
from sklearn import tree  #决策树构建所需的包
from sklearn.datasets import load_iris  #加载数据集
from sklearn.model_selection import GridSearchCV  #查找最优参数
#import pandas as pd
#from sklearn.externals.six import StringIO
#import pydot
#import pydotplus

if __name__ == '__main__':
    # show data info
    data = load_iris() # 加载 IRIS 数据集
    print('keys: \n', data.keys())
    feature_names = data.get('feature_names')
    target_names=data.get('target_names')
    print('feature names: \n', data.get('feature_names')) # 查看属性名称
    print('target names: \n', data.get('target_names')) # 查看 label 名称
    x = data.get('data') # 获取样本矩阵
    y = data.get('target') # 获取与样本对应的 label 向量
    print(x.shape, y.shape) # 查看样本数据
    #print(data)
    #print(target_names)

    # visualize the data
    f = []
    f.append(y == 0)  # 类别为第一类的样本的逻辑索引
    f.append(y == 1)  # 类别为第二类的样本的逻辑索引
    f.append(y == 2)  # 类别为第三类的样本的逻辑索引
    color = ['red', 'blue', 'green']
    fig, axes = plt.subplots(4, 4)  # 绘制四个属性两两之间的散点图
    for i, ax in enumerate(axes.flat):
        row = i // 4
        col = i % 4
        if row == col:
            ax.text(.1, .5, feature_names[row])
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for k in range(3):
            ax.scatter(x[f[k], row], x[f[k], col], c=color[k], s=3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 设置间距
    plt.show()

    # 随机划分训练集和测试集
    num = x.shape[0]  # 样本总数
    ratio = 7 / 3  # 划分比例，训练集数目:测试集数目
    num_test = int(num / (1 + ratio))  # 测试集样本数目
    num_train = num - num_test  # 训练集样本数目
    index = np.arange(num)  # 产生样本标号
    np.random.shuffle(index)  # 洗牌
    x_test = x[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
    y_test = y[index[:num_test]]
    x_train = x[index[num_test:], :]  # 剩余作为训练集
    y_train = y[index[num_test:]]

    entropy_thresholds = np.linspace(0, 1, 100)
    gini_thresholds = np.linspace(0, 0.2, 100)
    #设置参数矩阵：
    param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
                {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
                {'max_depth': np.arange(1,4)},
                {'min_samples_split': np.arange(2,30,1)}]
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5)
    clf.fit(x, y)
    print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))

    # 构建决策树
    clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=3)  # 建立决策树对象
    clf.fit(x_train, y_train)  # 决策树拟合
    print("train score:", clf.score(x_train, y_train))
    print("test score:", clf.score(x_test, y_test))

    # 预测
    y_test_pre = clf.predict(x_test)  # 利用拟合的决策树进行预测
    print('the predict values are', y_test_pre)  # 显示结果
    print('the trueaaa values are',y_test)

  # 计算分类准确率
    acc = sum(y_test_pre==y_test)/num_test
    print('the accuracy is', acc) # 显示预测准确率

  # 画出决策树并保存
    with open("allElectronicsData.dot", "w") as f:
        f = tree.export_graphviz(clf,  feature_names=feature_names,class_names=target_names,out_file=f)




