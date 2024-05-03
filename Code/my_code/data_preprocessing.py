import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 测试用工具类
import tools

# 导入数据集，X中保存特征，Y中保存标记
dataset = pd.read_csv('../../datasets/Data.csv')
# tools.print_dataframe(dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# 补全第二、三列的缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# 对离散数据进行编码
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# 将特征转换为独热编码
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
# 对标记进行编码
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 特征标准化
# 大部分模型算法使用两点间的欧氏距离表示，但此特征在幅度、单位和范围姿态嗯提上变化很大。
# 在距离计算中，高幅度的特征比低幅度特征权重更大。所以需要标准化。
# Standardization of datasets is a common requirement for many machine
# learning estimators implemented in scikit-learn; they might behave badly
# if the individual features do not more or less look like standard normally distributed data.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
