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
#一个驱动型prompt的例子
prompt_template = """You are an AI assistant that helps travelers pick a city to travel to. 
You do this by rating how much a person would enjoy a city based on their interests.
Given a city and interests, you respond with an integer 1-10 where 10 is the most enjoyment and 0 is the least.

Sample city: New York City
Sample interests: food, museums, hiking
Sample answer: 8

City: {city}
Interests: {interests}
Answer: """
#新的测试用例

response, pl_request_id = openai.completions.create(
  model="gpt-3.5-turbo-instruct", 
  prompt=prompt_template.format(city="Washington, D.C.", interests="resorts, museums, beaches"),
  #.format：格式化函数，
  pl_tags=["Test content01"],
  return_pl_id=True # Make sure to set this to True
)

#给定了一个正确答案的字典？

answer = response.choices[0].text
print(answer)

#对得到的答案进行检验和打分
numeric_answer = None
error_message = None
try:
    numeric_answer = int(answer.strip())
except ValueError as e:
    error_message = str(e)
    pass

# 使用 PromptLayer 中的分数来跟踪答案是否为 int
promptlayer.track.score(
    request_id=pl_request_id,
    score=50 if numeric_answer else 0,
)

print("Numeric answer:", numeric_answer)

#在 PromptLaye 中记录请求的元数据
promptlayer.track.metadata(
    request_id=pl_request_id,
    metadata={
        "referrer": "getting_started.ipynb",
        "origin": "NYC, USA",
        "user_id": "sdf328",
        "error_message": "No error" if numeric_answer else error_message,
    }
)

#现在已经在Pinecone中插入了基础数据
#需要根据GPT的思考对数据进行处理和识别


