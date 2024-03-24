#本data处理过程主要是处理引入prompt-work的data过程的

#####
#随机采样的覆盖率问题，样本之前的分组覆盖率是达不到的，随机的采样是不完善的。
#####

#主要需要处理：info文本的打包，松紧接口的判断，prompt逻辑顺序整理
from info_process import train_5_extract_info_by_number,train_minus_5_extract_info_by_number,test_extract_info_by_number,test_in_train_extract_info_by_number
from info_process import random_train_5_number,random_train_minus_5_number,random_test_number,random_test_in_train_number
#test_in_train真实值的调用
from info_process import  test_in_train_caps_judge_number
#需要注意到的是：random_test_number,random_test_in_train_number是一串数组
#引用其中的文件方法
gloval_i_for_count = 0
#这里应该传入一个收紧程度的值

# 调用松紧值回传的办法，初始值为0，第一次调用的时候返回为数值

#首先先def的函数是训练中传入的参数-->用于找到可能存在的OpenAI的优化方法
def info_prompt(relax_tension_num):
    #传入一计数指标i
    global gloval_i_for_count

    #读入随机数
    train_5_number_a,train_5_number_b = random_train_5_number()
    train_minus_5_number_a,train_minus_5_number_b = random_train_minus_5_number()
    test_in_train_rand_number = random_test_in_train_number()[gloval_i_for_count]
 
    #print(train_5_number_a,train_5_number_b)
    #print(train_minus_5_number_a,train_minus_5_number_b)
    #print(random_test_number())

    prompt_pre ='''
    你是一名人工智能医疗数据分析助手，医生判断病人的数据是否患病
    你可以通过以下的数据判断一个人是否换上了帕金森症状

    如果用-5~5之间的数据来衡量该测试患者的患病可能性，
    请你只是回答一个在-5~5之间的整数，其中-5是一定不患病，5是一定患病

    如果把你的判断条件的宽松或者是收紧指标作为指数进行判断，其中-5是过于宽松，5是过于收紧
    你上次的判断是-5，本次你判断的收紧宽松指标是{relax_tension}

    样本1：
    {train_example1}
    样本2：
    {train_example2}
    样本3：
    {train_example3}
    样本4：
    {train_example4}
    测试用例：
    {test_in_train_example}
    '''
    prompt_use = prompt_pre.format(
        #插入松弛指标
        relax_tension = relax_tension_num,
        #一个患病一个非患病的插入样本，第一个是患病
        train_example1 = train_5_extract_info_by_number(train_5_number_a),
        #第二个是非患病
        train_example2 = train_minus_5_extract_info_by_number(train_minus_5_number_a),
        #第三个是患病
        train_example3 = train_5_extract_info_by_number(train_5_number_b),
        #第四个是不患病
        train_example4 = train_minus_5_extract_info_by_number(train_minus_5_number_b),
        #测试用例的封装
        test_in_train_example =  test_in_train_extract_info_by_number(test_in_train_rand_number)
    )
    #print(prompt_use)

    gloval_i_for_count +=1
    #print("方法 a 被调用了，计数器值为:", gloval_i_for_count)
    print('打印全局变量的global的值')
    print(gloval_i_for_count)
    return(prompt_use,test_in_train_rand_number)

#看来只是传出来一个prompt远远不够
#需要触底出来跟随随机数出现的判断值是多少
#读入那个传到test_in_train里的参数值（本质上是数组的不放回随机数）
def judge_num_for_test_in_train_example(number):
    judge_number = test_in_train_caps_judge_number(number)
    return(judge_number)
