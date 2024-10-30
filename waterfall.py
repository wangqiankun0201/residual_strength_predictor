# 导库
import pandas as pd
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt


# 导入数据
data=pd.read_excel("all.xlsx")
# 读取数据特征和标签
X=data.iloc[:,0:-1]
y=data.iloc[:,-1]
# 实例化
scaler= StandardScaler() #实例化
scaler = scaler.fit(X) #fit，在这里本质是生成min(x)和max(x)
X = scaler.transform(X) #通过接口导出结果
X = pd.DataFrame(X)
# 数据集划分
X_train=X.iloc[0:2111]
X_test=X.iloc[2111:]
y_train=y.iloc[0:2111]
y_test=y.iloc[2111:]
lgb_m1 = LGBMRegressor(
    n_estimators=500,##
    num_leaves= 20,
    learning_rate=0.20337109665892747,##11
    colsample_bytree= 0.880185829848282,
    min_child_samples=1,##11
    reg_alpha=0,
    reg_lambda=10,
    random_state=42
).fit(X_train,y_train)
# 创建SHAP解释器
explainer = shap.TreeExplainer(lgb_m1)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)


#特征标签
feature_label=['Pipe diameter','Wall thickness','Ultimate tensile strength','Yield strength','Elastic modulus','Defect depth','Defect length','Defect width']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 13  # 设置字体大小为14
explanation=explainer(X_test)
shap.plots.waterfall(explanation[0])