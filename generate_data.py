# 生成测试数据
import numpy as np
import pandas as pd
# 训练集和验证集样本总个数
sample = 2000
train_data_path = 'train.csv'
validate_data_path = 'validate.csv'
predict_data_path = 'test.csv'

# 构造生成数据的模型
X1 = np.zeros((sample, 1))
X1[:, 0] = np.random.normal(1, 1, sample)
X2 = np.zeros((sample, 1))
X2[:, 0] = np.random.normal(2, 1, sample)
X3 = np.zeros((sample, 1))
X3[:, 0] = np.random.normal(3, 1, sample)

# 模型
Y = 6 * X1 - 3* X2 + X3 * X3 + np.random.normal(0, 0.1, [sample, 1])

# 将所有生成的数据放到data里面
data = np.zeros((sample, 4))
data[:, 0] = X1[:, 0]
data[:, 1] = X2[:, 0]
data[:, 2] = X3[:, 0]
data[:, 3] = Y[:, 0]

# 将data分成测试集和训练集
num_traindata = int(0.8*sample)

# 将训练数据保存
traindata = pd.DataFrame(data[0:num_traindata, :], columns=['x1', 'x2', 'x3', 'y'])
traindata.to_csv(train_data_path, index=False)
print('训练数据保存在: ', train_data_path)

# 将验证数据保存
validate_data = pd.DataFrame(data[num_traindata:, :], columns=['x1', 'x2', 'x3', 'y'])
validate_data.to_csv(validate_data_path, index=False)
print('验证数据保存在: ', validate_data_path)

# 将预测数据保存
predict_data = pd.DataFrame(data[num_traindata:, 0:-1], columns=['x1', 'x2', 'x3'])
predict_data.to_csv(predict_data_path, index=False)
print('预测数据保存在: ', predict_data_path)
