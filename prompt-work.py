'''
2024-3-23
畅畅子
'''


#根据算法流程图搭建一个提示工程
import os
from openai import OpenAI
import promptlayer
from dotenv import load_dotenv
#把prompt数据调用
from data_process import info_prompt,judge_num_for_test_in_train_example

load_dotenv('env01.env')
#到底GPT的底层有被调用吗？有的
promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
OpenAI = promptlayer.openai.OpenAI
openai = OpenAI()

#封装单次三次的训练过程
def cap_openai_judge(prompt_work_last,test_in_train_rand_number):
    response, pl_request_id = openai.completions.create(
        model="gpt-3.5-turbo-instruct", 
        prompt=prompt_work_last,
        max_tokens=500,  # 适当调整以避免截断，根据需要调整这个值
        temperature = 0.5,
        pl_tags=["Test for LLM2"],
        return_pl_id=True
    )
    answer = response.choices[0].text.strip()
    
    try:
        # 尝试将 answer 转换为整数
        int_answer = int(answer)
        # 如果 answer 是一个整数，则返回它
        #同时要return那个传入的test_in_train里的参数值，作为下列判断的选项
        return(int_answer,test_in_train_rand_number)
    except ValueError:
        # 如果 answer 不是一个整数，则重新调用函数 cap_openai_judge
        print("回答不是一个整数，重新调用函数")
        return cap_openai_judge(prompt_work_last,test_in_train_rand_number)


#封装数据处理的减法
#这里也是个比较困难的地方，在得到比较结果之后需要拿到test的原来判断值做abs

#判断的算法距离流程如下：
#num_judge1,num_judge2 = cap_openai_judge(prompt_work_last,test_in_train_rand_number)



def  calculate_status_num(num1,num2):
    original_judge = judge_num_for_test_in_train_example(num2)
    #openai的判断值-测试中的用例判断值
    caculate_singel = abs(num1 - original_judge)
    return caculate_singel

#得到单次小循环中的判断相减值
#single_small_circle_num = calculate_status_num(num_judge1,num_judge2)


#判断松弛值
def relax_tension_index(value):
    """
    根据输入值返回放松/紧张指数。
    
    :param value: 介于0到30之间的活动值。
    :return: 介于-5到5之间的整数，表示放松到紧张的指数。
    """
    if value < 0 or value > 30:
        raise ValueError("输入值必须在0到30之间。")
    
    if value <= 15:
        # 将0~15线性映射到-5~0
        return int(-5 + (value / 15) * 5)
    else:
        # 将15~30线性映射到0~5
        return int((value - 15) / 15 * 5)
#53个大循环中处理所有的test集合中的所有事件
    
#######
#正式开始本训练程序的运行
#######
#存储OpenAI接口返回的数值
OpenAI_result_list = []
#存储做出相减之后的值的数组
calculate_result_list = []
#储存收紧和放松值的数组
relax_tension_index_num_list = []

#需要及时的更新这里的信息函数

prompt_txt,test_in_train_rand_number = info_prompt(0)

for large_num in range(1,53):
    #3个小循环中处理松紧程度的反馈值
    #同时要处理读出数据
    #确定一个小循环的相加值
    sum_judge = 0 
    for i in range(1,4):
        #调取数字封装
        #这里需要修改-->调取的prompt_txt可以自调用变化随机数，但是我们的修正值要传进去
        num_judge1,num_judge2 = cap_openai_judge(prompt_txt,test_in_train_rand_number)
        #得到相减的值
        #将数据写入数组
        OpenAI_result_list.append(num_judge1)
        single_small_circle_num = calculate_status_num(num_judge1,num_judge2)
        #小轮次内的求和是三个求和
        sum_judge =   sum_judge +  single_small_circle_num
        #存储做出相减之后的值的数组
        calculate_result_list.append(single_small_circle_num)
        #每三个得到一个收紧和放松值
        #得到收紧和缩减值
    
    relax_tension_index_num = relax_tension_index(sum_judge)
    relax_tension_index_num_list.append( relax_tension_index_num )
    


    #内部需要写一个相加值，从for 内传出来进行判断

print('openai的返回数组') 
print(OpenAI_result_list)
print('存储做出相减之后的值的数组') 
print(calculate_result_list)
print('收缩指数数组') 
print(relax_tension_index_num_list) 
#大循环嵌套小循环
#195-39=156 156/3=52
#要打包一个OpenAI接口用来轮询的调取prompt和返回整数值（batch_size = 3）
#这个OpenAI接口是一次性处理一个，一次性返回判断一个，如果不是整数，就再次训练