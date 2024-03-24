import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("./parkinsons.data", sep=',', index_col='name')
dataset.head()
print(dataset.shape)
#相关性矩阵
corr = dataset.corr()
plt.figure(figsize=(24, 24))
sns.heatmap(corr, annot=True, cmap='coolwarm')

#plt.show()
#分离样本点和数据集
x = dataset.iloc[: , np.r_[0:16, 17:23]].values
y = dataset.iloc[: , 16].values

print(x.shape)




#分割数据集
#所有行数据是要洗牌过后的（打乱1/0排序）
#0.3作为测试数据集
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state= 1,stratify=y)
print(y_train)

#标准化处理数据
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#展示数据分类结果能力
def cm_displayer(cm):
  #从混淆矩阵创建一个 DataFrame。
  cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

  plt.figure(figsize=(10,7))

  sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')

  plt.title('Confusion Matrix')
  plt.show()

#开始随机森林进行分类

rf_classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=20)

rf_classifier.fit(x_train,y_train)

rf_model_filename = "RF_model.pkl"

with open(rf_model_filename,"wb") as file:
  pickle.dump(rf_classifier,file)


y_pred = rf_classifier.predict(x_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test,y_pred)

#打印分类的成功率

print(accuracy_score(y_test, y_pred))
cm_displayer(cm)

#随机森林模型能够提供关于特征重要性的直观了解。
#feature_importances = rf_classifier.feature_importances_

feature_names_left = dataset.columns[0:16]  # 选择前16列作为特征部分的名称
feature_names_right = dataset.columns[17:23]  # 跳过第16列，选择剩余列作为特征部分的名称
feature_names = feature_names_left.append(feature_names_right)  # 将两部分特征名称合并

# 根据随机森林模型的特征重要性，创建一个DataFrame
importances = pd.DataFrame({'feature': feature_names, 'importance': rf_classifier.feature_importances_})
# 按照特征重要性降序排列
importances = importances.sort_values('importance', ascending=False)

# 打印特征重要性
print(importances)