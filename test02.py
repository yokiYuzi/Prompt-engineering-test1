#正式的搭建一个从矢量数据库之中提取相似度数据的一算法
#主要涉及的是Pinecone + OpenAI + promptLayer
#但是比较存疑的问题有：矢量数据库的相似度算法能否对数据进行分类？
import os
from openai import OpenAI
import promptlayer
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('env01.env')
#到底GPT的底层有被调用吗？有的
promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
OpenAI = promptlayer.openai.OpenAI
openai = OpenAI()
Prompt_template = """你是一名人工智能医疗数据分析助手，医生判断病人的数据是否患病
你可以通过以下的数据判断一个人是否换上了帕金森症状

如果用-5~5之间的数据来衡量该测试患者的患病可能性，
请你只是回答一个在-5~5之间的整数，其中-5是一定不患病，5是一定患病

如果把你的判断条件的宽松或者是收紧指标作为指数进行判断，其中-5是过于宽松，5是过于收紧
你上次的判断是-5，请修改你的判断条件继续判断

sample1：
- 平均人声基频（MDVP:Fo(Hz)）为119.992赫兹。
- 最大人声基频（MDVP:Fhi(Hz)）为157.302赫兹。
- 最小人声基频（MDVP:Flo(Hz)）为74.997赫兹。
- 声音的整体抖动百分比（MDVP:Jitter(%)）为0.00784%。
- 抖动的绝对值（MDVP:Jitter(Abs)）为0.00007。
- 相对平均周期波动（MDVP:RAP）为0.0037。
- 相对周期波动（MDVP:PPQ）为0.00554。
- 三点抖动维度（Jitter:DDP）为0.01109。
- 声音的整体闪光（MDVP:Shimmer）为0.04374。
- 闪光的分贝值（MDVP:Shimmer(dB)）为0.426分贝。
- 三点闪光（Shimmer:APQ3）为0.02182。
- 五点闪光（Shimmer:APQ5）为0.0313。
- 总闪光（MDVP:APQ）为0.02971。
- 闪光的三倍维度（Shimmer:DDA）为0.06545。
- 噪声比率（NHR）为0.02211。
- 谐波与噪声比（HNR）为21.033。
- 语音信号的非线性动态复杂性（RPDE）为0.414783。
- 信号分形缩放指数（DFA）为0.815285。
- 第一个非线性动态复杂性指数（spread1）为-4.813031。
- 第二个非线性动态复杂性指数（spread2）为0.266482。
- 基频变化的非线性测量（D2）为2.301442。
- 声音振动的非线性测量（PPE）为0.284654。
问：该判断目标的患者的健康状况是什么？
回答：5

sample2：
-平均人声基频（MDVP:Fo(Hz)）为197.076赫兹。
-最大人声基频（MDVP:Fhi(Hz)）为206.896赫兹。
-最小人声基频（MDVP:Flo(Hz)）为192.055赫兹。
-声音的整体抖动百分比（MDVP:Jitter(%)）为0.00289%。
-抖动的绝对值（MDVP:Jitter(Abs)）为0.00001。
-相对平均周期波动（MDVP:RAP）为0.00166。
-相对周期波动（MDVP:PPQ）为0.00168。
-三点抖动维度（Jitter:DDP）为0.00498。
-声音的整体闪光（MDVP:Shimmer）为0.01098。
-闪光的分贝值（MDVP:Shimmer(dB)）为0.097分贝。
-三点闪光（Shimmer:APQ3）为0.00563。
-五点闪光（Shimmer:APQ5）为0.0068。
-总闪光（MDVP:APQ）为0.00802。
-闪光的三倍维度（Shimmer:DDA）为0.01689。
-噪声比率（NHR）为0.00339。
-谐波与噪声比（HNR）为26.775。
-语音信号的非线性动态复杂性（RPDE）为0.422229。
-信号分形缩放指数（DFA）为0.741367。
-第一个非线性动态复杂性指数（spread1）为-7.3483。
-第二个非线性动态复杂性指数（spread2）为0.177551。
-基频变化的非线性测量（D2）为1.743867。
-声音振动的非线性测量（PPE）为0.085569。
问：该判断目标的患者的健康状况是什么？
回答：-5


判断目标
- 平均人声基频（MDVP:Fo(Hz)）为{MDVP_Fo}赫兹。
- 最大人声基频（MDVP:Fhi(Hz)）为{MDVP_Fhi}赫兹。
- 最小人声基频（MDVP:Flo(Hz)）为{MDVP_Flo}赫兹。
- 声音的整体抖动百分比（MDVP:Jitter(%)）为{MDVP_Jitter_Percent}%。
- 抖动的绝对值（MDVP:Jitter(Abs)）为{MDVP_Jitter_Abs}。
- 相对平均周期波动（MDVP:RAP）为{MDVP_RAP}。
- 相对周期波动（MDVP:PPQ）为{MDVP_PPQ}。
- 三点抖动维度（Jitter:DDP）为{Jitter_DDP}。
- 声音的整体闪光（MDVP:Shimmer）为{MDVP_Shimmer}。
- 闪光的分贝值（MDVP:Shimmer(dB)）为{MDVP_Shimmer}分贝。
- 三点闪光（Shimmer:APQ3）为{Shimmer_APQ3}。
- 五点闪光（Shimmer:APQ5）为{Shimmer_APQ5}。
- 总闪光（MDVP:APQ）为{MDVP_APQ}。
- 闪光的三倍维度（Shimmer:DDA）为{Shimmer_DDA}。
- 噪声比率（NHR）为{NHR}。
- 谐波与噪声比（HNR）为{HNR}。
- 语音信号的非线性动态复杂性（RPDE）为{RPDE}。
- 信号分形缩放指数（DFA）为{DFA}。
- 第一个非线性动态复杂性指数（spread1）为{spread1}。
- 第二个非线性动态复杂性指数（spread2）为{spread2}。
- 基频变化的非线性测量（D2）为{D2}。
- 声音振动的非线性测量（PPE）为{PPE}。
问：该判断目标的患者的健康状况是什么？
回答：


"""
#是否要把病人的数据特征转化为对话信息-->这在微调中式需要的
#在prompt工程中是否需要类似的经验呢？
response, pl_request_id = openai.completions.create(
  model="gpt-3.5-turbo-instruct", 
  prompt=Prompt_template.format(
    MDVP_Fo= "126.344",
    MDVP_Fhi= "134.231",
    MDVP_Flo= "112.773",
    MDVP_Jitter_Percent= "0.00448",
    MDVP_Jitter_Abs= "0.00004",
    MDVP_RAP= "0.00131",
    MDVP_PPQ= "0.00169",
    Jitter_DDP= "0.00393",
    MDVP_Shimmer= "0.02033",
    MDVP_Shimmer_dB= "0.185",
    Shimmer_APQ3= "0.01143",
    Shimmer_APQ5= "0.00959",
    MDVP_APQ= "0.01614",
    Shimmer_DDA= "0.03429",
    NHR= "0.00474",
    HNR= "25.03",
    RPDE= "0.507504",
    DFA= "0.760361",
    spread1= "-6.689151",
    spread2= "0.291954",
    D2= "2.431854",
    PPE= "0.105993"
    ),
    #.format：格式化函数，
    #status= "患病" if 1 == 1 else "非患病"  # Assuming status is 1 for ill
    max_tokens=1500,  # 适当调整以避免截断，根据需要调整这个值
    temperature = 0.5,
    pl_tags=["Test for LLM1"],
    return_pl_id=True # Make sure to set this to True
)
#打印回答

answer = response.choices[0].text

print(answer)