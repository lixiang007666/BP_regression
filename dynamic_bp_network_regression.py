import tensorflow as tf
import pandas as pd
import numpy as np
createVar = locals()

'''
建立一个网络结构可变的BP神经网络通用代码：

在训练时各个参数的意义：
hidden_floors_num：隐藏层的个数
every_hidden_floor_num：每层隐藏层的神经元个数
learning_rate：学习速率
activation：激活函数
regularization：正则化方式
regularization_rate：正则化比率
total_step：总的训练次数
train_data_path：训练数据路径
model_save_path：模型保存路径

利用训练好的模型对验证集进行验证时各个参数的意义：
model_save_path：模型保存路径
validate_data_path：验证集路径
precision：精度

利用训练好的模型进行预测时各个参数的意义：
model_save_path：模型的保存路径
predict_data_path：预测数据路径
predict_result_save_path：预测结果保存路径
'''


# 训练模型全局参数
hidden_floors_num = 1
every_hidden_floor_num = [50]
learning_rate = 0.00001
activation = 'tanh'
regularization = 'L1'
regularization_rate = 0.0001
total_step = 100000
train_data_path = 'train.csv'
model_save_path = 'model/predict_model'

# 利用模型对验证集进行验证返回正确率
model_save_path = 'model/predict_model'
validate_data_path = 'validate.csv'
precision = 0.5

# 利用模型进行预测全局参数
model_save_path = 'model/predict_model'
predict_data_path = 'test.csv'
predict_result_save_path = 'test_predict.csv'


def inputs(train_data_path):
    train_data = pd.read_csv(train_data_path)
    X = np.array(train_data.iloc[:, :-1])
    Y = np.array(train_data.iloc[:, -1:])
    return X, Y


def make_hidden_layer(pre_lay_num, cur_lay_num, floor):
    createVar['w' + str(floor)] = tf.Variable(tf.random_normal([pre_lay_num, cur_lay_num], stddev=1))
    createVar['b' + str(floor)] = tf.Variable(tf.random_normal([cur_lay_num], stddev=1))
    return eval('w'+str(floor)), eval('b'+str(floor))


def initial_w_and_b(all_floors_num):
    # 初始化隐藏层的w, b
    for floor in range(2, hidden_floors_num+3):
        pre_lay_num = all_floors_num[floor-2]
        cur_lay_num = all_floors_num[floor-1]
        w_floor, b_floor = make_hidden_layer(pre_lay_num, cur_lay_num, floor)
        createVar['w' + str(floor)] = w_floor
        createVar['b' + str(floor)] = b_floor


def cal_floor_output(x, floor):
    w_floor = eval('w'+str(floor))
    b_floor = eval('b'+str(floor))
    if activation == 'sigmoid':
        output = tf.sigmoid(tf.matmul(x, w_floor) + b_floor)
    if activation == 'tanh':
        output = tf.tanh(tf.matmul(x, w_floor) + b_floor)
    if activation == 'relu':
        output = tf.nn.relu(tf.matmul(x, w_floor) + b_floor)
    return output


def inference(x):
    output = x
    for floor in range(2, hidden_floors_num+2):
        output = cal_floor_output(output, floor)

    floor = hidden_floors_num+2
    w_floor = eval('w'+str(floor))
    b_floor = eval('b'+str(floor))
    output = tf.matmul(output, w_floor) + b_floor
    return output


def loss(x, y_real):
    y_pre = inference(x)
    if regularization == 'None':
        total_loss = tf.reduce_sum(tf.squared_difference(y_real, y_pre))

    if regularization == 'L1':
        total_loss = 0
        for floor in range(2, hidden_floors_num + 3):
            w_floor = eval('w' + str(floor))
            total_loss = total_loss + tf.contrib.layers.l1_regularizer(regularization_rate)(w_floor)
        total_loss = total_loss + tf.reduce_sum(tf.squared_difference(y_real, y_pre))

    if regularization == 'L2':
        total_loss = 0
        for floor in range(2, hidden_floors_num + 3):
            w_floor = eval('w' + str(floor))
            total_loss = total_loss + tf.contrib.layers.l2_regularizer(regularization_rate)(w_floor)
        total_loss = total_loss + tf.reduce_sum(tf.squared_difference(y_real, y_pre))

    return total_loss


def train(total_loss):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    return train_op


# 训练模型
def train_model(hidden_floors_num, every_hidden_floor_num, learning_rate, activation, regularization,
                regularization_rate, total_step, train_data_path, model_save_path):
    file_handle = open('acc.txt', mode='w')
    X, Y = inputs(train_data_path)
    X_dim = X.shape[1]
    all_floors_num = [X_dim] + every_hidden_floor_num + [1]

    # 将参数保存到和model_save_path相同的文件夹下， 恢复模型进行预测时加载这些参数创建神经网络
    temp = model_save_path.split('/')
    model_name = temp[-1]
    parameter_path = ''
    for i in range(len(temp)-1):
        parameter_path = parameter_path + temp[i] + '/'
    parameter_path = parameter_path + model_name + '_parameter.txt'
    with open(parameter_path, 'w') as f:
        f.write("all_floors_num:")
        for i in all_floors_num:
            f.write(str(i) + ' ')
        f.write('\n')
        f.write('activation:')
        f.write(str(activation))

    x = tf.placeholder(dtype=tf.float32, shape=[None, X_dim])
    y_real = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    initial_w_and_b(all_floors_num)
    y_pre = inference(x)
    total_loss = loss(x, y_real)
    train_op = train(total_loss)

    # 记录在训练集上的正确率
    train_accuracy = tf.reduce_mean(tf.cast(tf.abs(y_pre - y_real) < precision, tf.float32))
    print(y_pre)
    # 保存模型
    saver = tf.train.Saver()

    # 在一个会话对象中启动数据流图，搭建流程
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(total_step):
        sess.run([train_op], feed_dict={x: X[0:, :], y_real: Y[0:, :]})
        if step % 1000 == 0:
            saver.save(sess, model_save_path)
            total_loss_value = sess.run(total_loss, feed_dict={x: X[0:, :], y_real: Y[0:, :]})
            lxacc=sess.run(train_accuracy, feed_dict={x: X, y_real: Y})
            print('train step is ', step, ', total loss value is ', total_loss_value,
                  ', train_accuracy', lxacc,
                  ', precision is ', precision)

            file_handle.write(str(lxacc)+"\n")


    saver.save(sess, model_save_path)
    sess.close()


def validate(model_save_path, validate_data_path, precision):
    # **********************根据model_save_path推出模型参数路径, 解析出all_floors_num和activation****************
    temp = model_save_path.split('/')
    model_name = temp[-1]
    parameter_path = ''
    for i in range(len(temp)-1):
        parameter_path = parameter_path + temp[i] + '/'
    parameter_path = parameter_path + model_name + '_parameter.txt'
    with open(parameter_path, 'r') as f:
        lines = f.readlines()

    # 从读取的内容中解析all_floors_num
    temp = lines[0].split(':')[-1].split(' ')
    all_floors_num = []
    for i in range(len(temp)-1):
        all_floors_num = all_floors_num + [int(temp[i])]

    # 从读取的内容中解析activation
    activation = lines[1].split(':')[-1]
    hidden_floors_num = len(all_floors_num) - 2

    # **********************读取验证数据*************************************
    X, Y = inputs(validate_data_path)
    X_dim = X.shape[1]

    # **********************创建神经网络************************************
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_dim])
    y_real = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    initial_w_and_b(all_floors_num)
    y_pre = inference(x)

    # 记录在验证集上的正确率
    validate_accuracy = tf.reduce_mean(tf.cast(tf.abs(y_pre - y_real) < precision, tf.float32))

    sess = tf.Session()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 读取模型
        try:
            saver.restore(sess, model_save_path)
            print('模型载入成功！')
        except:
            print('模型不存在，请先训练模型！')
            return
        validate_accuracy_value = sess.run(validate_accuracy, feed_dict={x: X, y_real: Y})
        print('validate_accuracy is ', validate_accuracy_value)

    return validate_accuracy_value


def predict(model_save_path, predict_data_path, predict_result_save_path):
    # **********************根据model_save_path推出模型参数路径, 解析出all_floors_num和activation****************
    temp = model_save_path.split('/')
    model_name = temp[-1]
    parameter_path = ''
    for i in range(len(temp)-1):
        parameter_path = parameter_path + temp[i] + '/'
    parameter_path = parameter_path + model_name + '_parameter.txt'
    with open(parameter_path, 'r') as f:
        lines = f.readlines()

    # 从读取的内容中解析all_floors_num
    temp = lines[0].split(':')[-1].split(' ')
    all_floors_num = []
    for i in range(len(temp)-1):
        all_floors_num = all_floors_num + [int(temp[i])]

    # 从读取的内容中解析activation
    activation = lines[1].split(':')[-1]
    hidden_floors_num = len(all_floors_num) - 2

    # **********************读取预测数据*************************************
    predict_data = pd.read_csv(predict_data_path)
    X = np.array(predict_data.iloc[:, :])
    X_dim = X.shape[1]

    # **********************创建神经网络************************************
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_dim])
    initial_w_and_b(all_floors_num)
    y_pre = inference(x)

    sess = tf.Session()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 读取模型
        try:
            saver.restore(sess, model_save_path)
            print('模型载入成功！')
        except:
            print('模型不存在，请先训练模型！')
            return
        y_pre_value = sess.run(y_pre, feed_dict={x: X[0:, :]})

        # 将预测结果写入csv文件
        predict_data_columns = list(predict_data.columns) + ['predict']
        data = np.column_stack([X, y_pre_value])
        result = pd.DataFrame(data, columns=predict_data_columns)
        result.to_csv(predict_result_save_path, index=False)
        print('预测结果保存在：', predict_result_save_path)


if __name__ == '__main__':
    mode = "predict"

    if mode == 'train':
        # 训练模型
        train_model(hidden_floors_num, every_hidden_floor_num, learning_rate, activation, regularization,
                    regularization_rate, total_step, train_data_path, model_save_path)

    if mode == 'validate':
        # 利用模型对验证集进行正确性测试
        validate(model_save_path, validate_data_path, precision)

    if mode == 'predict':
        # 利用模型进行预测
        predict(model_save_path, predict_data_path, predict_result_save_path)












