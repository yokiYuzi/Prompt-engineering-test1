确定一更完善的流程来调取和思考
1.	首先要思考的是：数据的插入和训练的格式，即要回答以下问题：
1)	单次插入多少数据作为prompt比较合理？（在有限的4096 token条件下插入多少能够成为一个准确的参考标准？）

先插入一个3*3的6连续prompt测试
		
2)	Transformer的构架连续性问题：当我有多个对话窗口之后，得到的一系列的判断的对话点是否还是准确的？上下文的注意力机制的权重比率是多高？
3)	我能否自主的训练一个很好的判断模型呢？这里可以做很多判断的事情（先不用进行数据权重的筛选实验现在先使用所有的数据进行探测）

由此为思考搭建一个从神经网络思想取材的一个prompt训练工程与调试工程


![image](https://github.com/yokiYuzi/Prompt-engineering-test1/assets/76743561/eb593156-df08-482b-840e-df6360b8e353)


设计的范例样本如下：
![image](https://github.com/yokiYuzi/Prompt-engineering-test1/assets/76743561/3b1e13f4-2a26-432b-aab5-820153c5756c)

![image](https://github.com/yokiYuzi/Prompt-engineering-test1/assets/76743561/5613dad5-372d-49bc-aa72-1ca2d2b72437)

训练目标如下：
![image](https://github.com/yokiYuzi/Prompt-engineering-test1/assets/76743561/cc8ab5e1-a3fe-417d-86f0-efd9bc029a75)


封装流程：
![image](https://github.com/yokiYuzi/Prompt-engineering-test1/assets/76743561/f787eb35-5e81-4bc8-a344-6b01366ffd6f)

设计的Main训练方式：
![image](https://github.com/yokiYuzi/Prompt-engineering-test1/assets/76743561/45edc89e-69ee-4f26-a074-1ca28287a777)
