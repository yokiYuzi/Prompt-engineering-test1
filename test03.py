from data_process import info_prompt

for i in range(1,53):
    for i in range(1,4):
        prompt_txt,test_in_train_rand_number = info_prompt(0)
    #print(prompt_txt)
        print('打印train训练中的随机number')
        print(test_in_train_rand_number)

