import pandas as pd
import numpy as np

#载入数据
file_path = './pre-data.csv'
data = pd.read_csv(file_path)
#查看文件头
#print(data.head())



#data中的随机数据列为number
data['number'] = np.arange(1, len(data) + 1)
# 把随机数字列存入status前
#, random_state=42
shuffled_numbers = data['number'].sample(frac=1).reset_index(drop=True)
data['number'] = shuffled_numbers
# 变更衡量标准-->采用一个区间形式来表示可能患病的程度
data['status'] = data['status'].apply(lambda x: 5 if x == 1 else -5)
#看一下文件格式
#print(data[['name', 'number', 'status']].head())

# 第一步：统计并排序
status_5_data = data[data['status'] == 5].sort_values(by='number')
status_minus_5_data = data[data['status'] == -5].sort_values(by='number')
#打印为5与-5的样本和
#print(status_5_data)
#print(status_minus_5_data)

# 第二步：分割数据集
train_5 = status_5_data.sample(frac=0.8, random_state=42)
test_5 = status_5_data.drop(train_5.index)
train_minus_5 = status_minus_5_data.sample(frac=0.8, random_state=42)
test_minus_5 = status_minus_5_data.drop(train_minus_5.index)
#对number列重新排序
# 对于训练集和测试集中状态为5的数据
train_5['number'] = np.arange(1, len(train_5) + 1)
test_5['number'] = np.arange(1, len(test_5) + 1)

# 对于训练集和测试集中状态为-5的数据
train_minus_5['number'] = np.arange(1, len(train_minus_5) + 1)
test_minus_5['number'] = np.arange(1, len(test_minus_5) + 1)
#整理test集合为一个集合

test_set = pd.concat([test_5, test_minus_5])
#洗牌过的数据
test_set_shuffled = test_set.sample(frac=1, random_state=42).reset_index(drop=True)
#对number列进行重新排序
test_set_shuffled['number'] = np.arange(1, len(test_set_shuffled) + 1)
# 打印test_set_shuffled的前几行以确认结果

#也需要做一个合并的train_set
train_set = pd.concat([train_5,train_minus_5])
#洗牌过的数据
train_set_shuffled = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
#对number列进行重新排序
train_set_shuffled['number'] = np.arange(1, len(train_set_shuffled) + 1)


#对数据进行prompt处理
#一共有48个-5，147个5



#训练的样本数据（回答中作为样本存在的数据）

def train_5_extract_info_by_number(number):
    #找到对应行的随机数
    row = train_5[train_5['number'] == number]
    
    # Check if the row is found
    if row.empty:
        return "No data found for the given number."
    
    #插入行对应值
    info = f"""
    - 平均人声基频（MDVP:Fo(Hz)）为{row['MDVP:Fo(Hz)'].values[0]}赫兹 
    - 最大人声基频（MDVP:Fhi(Hz)）为{row['MDVP:Fhi(Hz)'].values[0]}赫兹 
    - 最小人声基频（MDVP:Flo(Hz)）为{row['MDVP:Flo(Hz)'].values[0]}赫兹 
    - 声音的整体抖动百分比（MDVP:Jitter(%)）为{row['MDVP:Jitter(%)'].values[0]}% 
    - 抖动的绝对值（MDVP:Jitter(Abs)）为{row['MDVP:Jitter(Abs)'].values[0]} 
    - 相对平均周期波动（MDVP:RAP）为{row['MDVP:RAP'].values[0]} 
    - 相对周期波动（MDVP:PPQ）为{row['MDVP:PPQ'].values[0]} 
    - 三点抖动维度（Jitter:DDP）为{row['Jitter:DDP'].values[0]} 
    - 声音的整体闪光（MDVP:Shimmer）为{row['MDVP:Shimmer'].values[0]} 
    - 闪光的分贝值（MDVP:Shimmer(dB)）为{row['MDVP:Shimmer(dB)'].values[0]}分贝 
    - 三点闪光（Shimmer:APQ3）为{row['Shimmer:APQ3'].values[0]} 
    - 五点闪光（Shimmer:APQ5）为{row['Shimmer:APQ5'].values[0]} 
    - 总闪光（MDVP:APQ）为{row['MDVP:APQ'].values[0]} 
    - 闪光的三倍维度（Shimmer:DDA）为{row['Shimmer:DDA'].values[0]} 
    - 噪声比率（NHR）为{row['NHR'].values[0]} 
    - 谐波与噪声比（HNR）为{row['HNR'].values[0]} 
    - 语音信号的非线性动态复杂性（RPDE）为{row['RPDE'].values[0]} 
    - 信号分形缩放指数（DFA）为{row['DFA'].values[0]} 
    - 第一个非线性动态复杂性指数（spread1）为{row['spread1'].values[0]} 
    - 第二个非线性动态复杂性指数（spread2）为{row['spread2'].values[0]} 
    - 基频变化的非线性测量（D2）为{row['D2'].values[0]} 
    - 声音振动的非线性测量（PPE）为{row['PPE'].values[0]} 
    问：该判断目标的患者的健康状况是什么？
    回答：{row['status'].values[0]} 
    """
    #打印一下info
    #print(info)
    return info

def train_minus_5_extract_info_by_number(number):
    #找到对应行的随机数
    row = train_minus_5[train_minus_5['number'] == number]
    
    # Check if the row is found
    if row.empty:
        return "No data found for the given number."
    
    #插入行对应值
    info = f"""
    - 平均人声基频（MDVP:Fo(Hz)）为{row['MDVP:Fo(Hz)'].values[0]}赫兹 
    - 最大人声基频（MDVP:Fhi(Hz)）为{row['MDVP:Fhi(Hz)'].values[0]}赫兹 
    - 最小人声基频（MDVP:Flo(Hz)）为{row['MDVP:Flo(Hz)'].values[0]}赫兹 
    - 声音的整体抖动百分比（MDVP:Jitter(%)）为{row['MDVP:Jitter(%)'].values[0]}% 
    - 抖动的绝对值（MDVP:Jitter(Abs)）为{row['MDVP:Jitter(Abs)'].values[0]} 
    - 相对平均周期波动（MDVP:RAP）为{row['MDVP:RAP'].values[0]} 
    - 相对周期波动（MDVP:PPQ）为{row['MDVP:PPQ'].values[0]} 
    - 三点抖动维度（Jitter:DDP）为{row['Jitter:DDP'].values[0]} 
    - 声音的整体闪光（MDVP:Shimmer）为{row['MDVP:Shimmer'].values[0]} 
    - 闪光的分贝值（MDVP:Shimmer(dB)）为{row['MDVP:Shimmer(dB)'].values[0]}分贝 
    - 三点闪光（Shimmer:APQ3）为{row['Shimmer:APQ3'].values[0]} 
    - 五点闪光（Shimmer:APQ5）为{row['Shimmer:APQ5'].values[0]} 
    - 总闪光（MDVP:APQ）为{row['MDVP:APQ'].values[0]} 
    - 闪光的三倍维度（Shimmer:DDA）为{row['Shimmer:DDA'].values[0]} 
    - 噪声比率（NHR）为{row['NHR'].values[0]} 
    - 谐波与噪声比（HNR）为{row['HNR'].values[0]} 
    - 语音信号的非线性动态复杂性（RPDE）为{row['RPDE'].values[0]} 
    - 信号分形缩放指数（DFA）为{row['DFA'].values[0]} 
    - 第一个非线性动态复杂性指数（spread1）为{row['spread1'].values[0]} 
    - 第二个非线性动态复杂性指数（spread2）为{row['spread2'].values[0]} 
    - 基频变化的非线性测量（D2）为{row['D2'].values[0]} 
    - 声音振动的非线性测量（PPE）为{row['PPE'].values[0]} 
    问：该判断目标的患者的健康状况是什么？
    回答：{row['status'].values[0]} 
    """
    #打印一下info
    #print(info)
    return info

def train_extract_info_by_number(number):
    #找到对应行的随机数
    row = data[data['number'] == number]
    
    # Check if the row is found
    if row.empty:
        return "No data found for the given number."
    
    #插入行对应值
    info = f"""
    - 平均人声基频（MDVP:Fo(Hz)）为{row['MDVP:Fo(Hz)'].values[0]}赫兹 
    - 最大人声基频（MDVP:Fhi(Hz)）为{row['MDVP:Fhi(Hz)'].values[0]}赫兹 
    - 最小人声基频（MDVP:Flo(Hz)）为{row['MDVP:Flo(Hz)'].values[0]}赫兹 
    - 声音的整体抖动百分比（MDVP:Jitter(%)）为{row['MDVP:Jitter(%)'].values[0]}% 
    - 抖动的绝对值（MDVP:Jitter(Abs)）为{row['MDVP:Jitter(Abs)'].values[0]} 
    - 相对平均周期波动（MDVP:RAP）为{row['MDVP:RAP'].values[0]} 
    - 相对周期波动（MDVP:PPQ）为{row['MDVP:PPQ'].values[0]} 
    - 三点抖动维度（Jitter:DDP）为{row['Jitter:DDP'].values[0]} 
    - 声音的整体闪光（MDVP:Shimmer）为{row['MDVP:Shimmer'].values[0]} 
    - 闪光的分贝值（MDVP:Shimmer(dB)）为{row['MDVP:Shimmer(dB)'].values[0]}分贝 
    - 三点闪光（Shimmer:APQ3）为{row['Shimmer:APQ3'].values[0]} 
    - 五点闪光（Shimmer:APQ5）为{row['Shimmer:APQ5'].values[0]} 
    - 总闪光（MDVP:APQ）为{row['MDVP:APQ'].values[0]} 
    - 闪光的三倍维度（Shimmer:DDA）为{row['Shimmer:DDA'].values[0]} 
    - 噪声比率（NHR）为{row['NHR'].values[0]} 
    - 谐波与噪声比（HNR）为{row['HNR'].values[0]} 
    - 语音信号的非线性动态复杂性（RPDE）为{row['RPDE'].values[0]} 
    - 信号分形缩放指数（DFA）为{row['DFA'].values[0]} 
    - 第一个非线性动态复杂性指数（spread1）为{row['spread1'].values[0]} 
    - 第二个非线性动态复杂性指数（spread2）为{row['spread2'].values[0]} 
    - 基频变化的非线性测量（D2）为{row['D2'].values[0]} 
    - 声音振动的非线性测量（PPE）为{row['PPE'].values[0]} 
    问：该判断目标的患者的健康状况是什么？
    回答：{row['status'].values[0]} 
    """
    #打印一下info
    #print(info)
    return info


#训练中需要回答的数据
#训练中需要回答的数据设计为不放回形式的抽样
def test_in_train_extract_info_by_number(number):
    #找到对应行的随机数
    row = train_set_shuffled[train_set_shuffled['number'] == number]
    
    # Check if the row is found
    if row.empty:
        return "No data found for the given number."
    
    #插入行对应值
    info = f"""
    - 平均人声基频（MDVP:Fo(Hz)）为{row['MDVP:Fo(Hz)'].values[0]}赫兹 
    - 最大人声基频（MDVP:Fhi(Hz)）为{row['MDVP:Fhi(Hz)'].values[0]}赫兹 
    - 最小人声基频（MDVP:Flo(Hz)）为{row['MDVP:Flo(Hz)'].values[0]}赫兹 
    - 声音的整体抖动百分比（MDVP:Jitter(%)）为{row['MDVP:Jitter(%)'].values[0]}% 
    - 抖动的绝对值（MDVP:Jitter(Abs)）为{row['MDVP:Jitter(Abs)'].values[0]} 
    - 相对平均周期波动（MDVP:RAP）为{row['MDVP:RAP'].values[0]} 
    - 相对周期波动（MDVP:PPQ）为{row['MDVP:PPQ'].values[0]} 
    - 三点抖动维度（Jitter:DDP）为{row['Jitter:DDP'].values[0]} 
    - 声音的整体闪光（MDVP:Shimmer）为{row['MDVP:Shimmer'].values[0]} 
    - 闪光的分贝值（MDVP:Shimmer(dB)）为{row['MDVP:Shimmer(dB)'].values[0]}分贝 
    - 三点闪光（Shimmer:APQ3）为{row['Shimmer:APQ3'].values[0]} 
    - 五点闪光（Shimmer:APQ5）为{row['Shimmer:APQ5'].values[0]} 
    - 总闪光（MDVP:APQ）为{row['MDVP:APQ'].values[0]} 
    - 闪光的三倍维度（Shimmer:DDA）为{row['Shimmer:DDA'].values[0]} 
    - 噪声比率（NHR）为{row['NHR'].values[0]} 
    - 谐波与噪声比（HNR）为{row['HNR'].values[0]} 
    - 语音信号的非线性动态复杂性（RPDE）为{row['RPDE'].values[0]} 
    - 信号分形缩放指数（DFA）为{row['DFA'].values[0]} 
    - 第一个非线性动态复杂性指数（spread1）为{row['spread1'].values[0]} 
    - 第二个非线性动态复杂性指数（spread2）为{row['spread2'].values[0]} 
    - 基频变化的非线性测量（D2）为{row['D2'].values[0]} 
    - 声音振动的非线性测量（PPE）为{row['PPE'].values[0]} 
    问：该判断目标的患者的健康状况是什么？
    回答：
    """
    #打印一下info
    #print(info)
    return info


#需要回答的数据
#需要回答的数据设计为不放回形式的抽样
def test_extract_info_by_number(number):
    #找到对应行的随机数
    row = test_set_shuffled[test_set_shuffled['number'] == number]
    
    # Check if the row is found
    if row.empty:
        return "No data found for the given number."
    
    #插入行对应值
    info = f"""
    - 平均人声基频（MDVP:Fo(Hz)）为{row['MDVP:Fo(Hz)'].values[0]}赫兹 
    - 最大人声基频（MDVP:Fhi(Hz)）为{row['MDVP:Fhi(Hz)'].values[0]}赫兹 
    - 最小人声基频（MDVP:Flo(Hz)）为{row['MDVP:Flo(Hz)'].values[0]}赫兹 
    - 声音的整体抖动百分比（MDVP:Jitter(%)）为{row['MDVP:Jitter(%)'].values[0]}% 
    - 抖动的绝对值（MDVP:Jitter(Abs)）为{row['MDVP:Jitter(Abs)'].values[0]} 
    - 相对平均周期波动（MDVP:RAP）为{row['MDVP:RAP'].values[0]} 
    - 相对周期波动（MDVP:PPQ）为{row['MDVP:PPQ'].values[0]} 
    - 三点抖动维度（Jitter:DDP）为{row['Jitter:DDP'].values[0]} 
    - 声音的整体闪光（MDVP:Shimmer）为{row['MDVP:Shimmer'].values[0]} 
    - 闪光的分贝值（MDVP:Shimmer(dB)）为{row['MDVP:Shimmer(dB)'].values[0]}分贝 
    - 三点闪光（Shimmer:APQ3）为{row['Shimmer:APQ3'].values[0]} 
    - 五点闪光（Shimmer:APQ5）为{row['Shimmer:APQ5'].values[0]} 
    - 总闪光（MDVP:APQ）为{row['MDVP:APQ'].values[0]} 
    - 闪光的三倍维度（Shimmer:DDA）为{row['Shimmer:DDA'].values[0]} 
    - 噪声比率（NHR）为{row['NHR'].values[0]} 
    - 谐波与噪声比（HNR）为{row['HNR'].values[0]} 
    - 语音信号的非线性动态复杂性（RPDE）为{row['RPDE'].values[0]} 
    - 信号分形缩放指数（DFA）为{row['DFA'].values[0]} 
    - 第一个非线性动态复杂性指数（spread1）为{row['spread1'].values[0]} 
    - 第二个非线性动态复杂性指数（spread2）为{row['spread2'].values[0]} 
    - 基频变化的非线性测量（D2）为{row['D2'].values[0]} 
    - 声音振动的非线性测量（PPE）为{row['PPE'].values[0]} 
    问：该判断目标的患者的健康状况是什么？
    回答：
    """
    #打印一下info
    #print(info)
    return info
#调取尝试

#创建一个可以返回test_in_train的真实判断值的方法
def test_in_train_caps_judge_number(number):
    row = train_set_shuffled[train_set_shuffled['number'] == number]
    return row['status'].values[0]

#train_5_extract_info_by_number(5)
#train_minus_5_extract_info_by_number(5)

#195-39=156 156/3=52
#也就是说单次的prompt训练有52batch,batch_size=3

#放回式的抽取5和-5的train值
def random_train_5_number():
    sampled_train_5_numbers = train_5.sample(n=2, replace=True)['number'].values
    #print(sampled_train_5_numbers)
    return(sampled_train_5_numbers)

def random_train_minus_5_number():
    sampled_train_minus_5_numbers = train_minus_5.sample(n=2, replace=True)['number'].values
    #print(sampled_train_minus_5_numbers)
    return(sampled_train_minus_5_numbers)


#不放回的抽取test_set
def random_test_number():
    len_test_idnex = len(test_set_shuffled)
    #print(len_test_idnex)
    #print(test_set_shuffled['number'].values)
    sampled_test_set_shuffled_number = test_set_shuffled.sample(n=len_test_idnex, replace=False,random_state=777)['number'].values
    #print(sampled_test_set_shuffled_number)
    return(sampled_test_set_shuffled_number)


#这里也要设定不放回的test_in_train的随机number
#195-39=156 156/3=52
#,random_state=777
def random_test_in_train_number():
    len_test_idnex = len(train_set_shuffled)
    #print(len_test_idnex)
    #print(test_set_shuffled['number'].values)
    sampled_test_in_train_set_shuffled_number = train_set_shuffled.sample(n=len_test_idnex, replace=False)['number'].values
    #print(sampled_test_in_train_set_shuffled_number)
    #正确的，找到了156
    return(sampled_test_in_train_set_shuffled_number)


#test_in_train_extract_info_by_number(5)
#test_extract_info_by_number(5)
#random_test_in_train_number()

