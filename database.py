from pinecone import Pinecone, ServerlessSpec
import pandas as pd

pc = Pinecone(api_key="1326fa61-5b44-4f58-9db3-a5ce56a6f21e")
#创建新的index
index = pc.Index("test01")
# 读取数据
data_path = "./parkinsons.data"
df = pd.read_csv(data_path)

# 处理数据并插入到Pinecone
# 假设所有字段除了'name'都是数值型，并将被作为向量的元素
#数据其实可以按照state来分类：1 （患病） 0（健康）
for _, row in df.iterrows():
    vector_id = row['name']
    # 将所有元素转换为浮点数
    vector = row.drop('name').astype(float).values.tolist()
    index.upsert(vectors=[(vector_id, vector)])

print("数据插入完成。")