
"""

多模态跨被试









"""



from timm.models.convnext import convnext_small
from torch.optim import lr_scheduler


#测试功能
def function_testData():
    import torch
    import numpy as np
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    #读取训练数据

    TrainLabelData = np.load('Data/Raw_Data_npy/train_label.npy').tolist()
    TrainLabel = []
    for i in TrainLabelData:
        if i[0] < 0.35:
            TrainLabel.append(0)
        else:
            TrainLabel.append(1)
    TrainLabelTensor = torch.tensor(TrainLabel)

    #读取测试数据
    TestLabelData = np.load('Data/Raw_Data_npy/test_label.npy').tolist()
    TestLabel = []
    for i in TestLabelData:
        if i[0] < 0.35:
            TestLabel.append(0)
        else:
            TestLabel.append(1)

    TestLabelTensor = torch.tensor(TestLabel)


    label = np.load('Data/label_npy/label_all.npy')
    TrainLabelAllData = label.tolist()
    TrainAllLabel = []
    for i in TrainLabelAllData:
        if i[0] < 0.35:
            TrainAllLabel.append(0)
        else:
            TrainAllLabel.append(1)
    TrainLabelTensor = torch.tensor(TrainLabel)








    return 0


# 读取眼电与脑电数据(OK)
def function_readData(index):
    import torch
    import numpy as np
    import os
    from scipy.io import loadmat
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 读取顺序表
    table = [[[1], [2, 3, 4, 5]], [[2], [1, 3, 4, 5]], [[3], [1, 2, 4, 5]], [[4], [1, 2, 3, 5]], [[5], [1, 2, 3, 4]]]

    # 读取训练数据
    eegDETrain = torch.empty([0, 17, 16, 5])
    eogDETrain = torch.empty([0, 3, 36])
    labelTrain = torch.empty([0, 1])
    for i in table[index - 1][1]:
        eegData = np.load('Data/dataset_DE/5zhejiaocha/' + str(i) + '.3features_npy')
        eogData = np.load('SEED-VIG/EOG_Feature/3features_npy/5zhejiaocha/' + str(i) + '.3features_npy')
        label = np.load('Data/label_npy/5zhejiaocha/' + str(i) + '.3features_npy')

        eegDataTensor = torch.tensor(eegData)
        eogDataTensor = torch.tensor(eogData)
        labelTensor = torch.tensor(label)
        eegDETrain = torch.cat([eegDETrain, eegDataTensor], dim=0)
        eogDETrain = torch.cat([eogDETrain, eogDataTensor], dim=0)
        labelTrain = torch.cat([labelTrain, labelTensor], dim=0)
    TrainLabelData = labelTrain.tolist()
    TrainLabel = []
    for i in TrainLabelData:
        if i[0] < 0.35:
            TrainLabel.append(0)
        else:
            TrainLabel.append(1)
    TrainLabelTensor = torch.tensor(TrainLabel)


    # 读取测试数据
    eegTestDataTensor = torch.tensor(np.load('Data/dataset_DE/5zhejiaocha/' + str(table[index - 1][0][0]) + '.3features_npy'))
    eogTestDataTensor = torch.tensor(np.load('SEED-VIG/EOG_Feature/3features_npy/5zhejiaocha/' + str(table[index - 1][0][0]) + '.3features_npy'))
    TestLabelData = np.load('Data/label_npy/5zhejiaocha/' + str(table[index - 1][0][0]) + '.3features_npy').tolist()
    TestLabel = []
    for i in TestLabelData:
        if i[0] < 0.35:
            TestLabel.append(0)
        else:
            TestLabel.append(1)

    TestLabelTensor = torch.tensor(TestLabel)

    return eegDETrain,eogDETrain, TrainLabelTensor, eegTestDataTensor,eogTestDataTensor, TestLabelTensor






#将数据放入模型测试
def function_three(model_i):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import time
    from pytorchtools import EarlyStopping
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.nn import Linear
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    start = time.time()

    # 模型超参数
    Batch_Size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据并打乱
    indexs = [1, 2, 3, 4, 5]
    accuracyMaxLists = []
    for index in indexs:
        print("__________________第{}次交叉验证__________________".format(index))
        # 读取数据
        eegDETrainTensor, eogDETrainTensor,TrainLabelTensor, eegDETestTensor,eogTestDataTensor, TestLabelTensor = function_readData(index)
        eegDETrainTensor = eegDETrainTensor.view(14160, 17, -1)
        eegDETestTensor = eegDETestTensor.view(3540, 17, -1)



        train_loader = DataLoader(dataset=TensorDataset(eegDETrainTensor,eogDETrainTensor, TrainLabelTensor), batch_size=Batch_Size,
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset=TensorDataset(eegDETestTensor,eogTestDataTensor,TestLabelTensor), batch_size=Batch_Size,
                                 shuffle=True, drop_last=True)


        from IENET_Model_Seven_3_eeg_eog_DE import MultimodalModelOne

        if model_i == 1:
            model = MultimodalModelOne().to(device)



        # 给定优化器:加了L2正则化;momentum,在SGD的基础上加一个动量，如果当前收敛效果好，就可以加速收敛，如果不好，则减慢它的步伐。
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.3)
        # # 定义学习率与轮数关系的函数
        lambda1 = lambda epoch: 0.99 ** epoch  # 学习率 = 0.95**(轮数)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # optimizer_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # 损失函数
        loss_fn = torch.nn.CrossEntropyLoss()
        # 初始化 early_stopping 对象
        patience = 40  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容
        # 相关训练数据
        EPOCH = 100
        epoch_list = []
        lossTrain_list = []
        lossTest_list = []
        accuracy_list = []
        lr_list = []
        # 一个计数器:只输出最后一次训练集的数据
        count_train = 0
        count_test = 0

        for epoch in range(EPOCH):
            epoch_list.append(epoch + 1)
            for eeg,eog, targets in train_loader:
                output = model(eeg.to(torch.float32).view(eeg.shape[0],17,80).to(device),eog.to(torch.float32).view(eog.shape[0],3,36).to(device))
                loss_train = loss_fn(output, targets.to(torch.long).to(device))
                optimizer.zero_grad()  # 优化之前必备工作：梯度清零
                loss_train.backward()  # 后向传播求导
                optimizer.step()  # 优化参数

                count_train += 1
                if count_train % 442 == 0:
                    print("第{}次训练的损失值(训练集)：{}".format(epoch + 1, loss_train.item()))
                    lossTrain_list.append(loss_train.item())
                    count_train = 0

            with torch.no_grad():  # 将模型的梯度清零，进行测试
                number_right = 0
                for eeg_test,eog_test, targets_test in test_loader:
                    outputs_test = model(eeg_test.to(torch.float32).view(eeg_test.shape[0],17,80).to(device),eog_test.to(torch.float32).view(eog_test.shape[0],3,36).to(device))
                    loss_test = loss_fn(outputs_test, targets_test.to(torch.long).to(device))
                    number_right += (outputs_test.argmax(1) == targets_test.to(device)).sum().item()  # 统计每次测试集合中，正确的个数
                    count_test += 1
                    if count_test % 110 == 0:
                        print("第{}次训练的损失值(测试集)：{}".format(epoch + 1, loss_test.item()))
                        lossTest_list.append(loss_test.item())
                accuracy = number_right / 3520
                print("第{}次整体测试集上的正确率:{}".format(epoch + 1, accuracy))
                accuracy_list.append(accuracy)

            # 学习率衰减
            scheduler.step()
            print("epoch={}, lr={}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            # 画图时让x与y有相同维度
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                break;



        now = time.time()
        print("本次训练花费时间：{}s".format(now - start))
        print(accuracy_list)
        print(max(accuracy_list))
        file_handle = open("Result/10.txt", mode='a+')
        file_handle.write(str(max(accuracy_list)) + "\t")
        file_handle.close()
        accuracyMaxLists.append(max(accuracy_list))

    return accuracyMaxLists








def function_test():
    # 解决bug
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Linear
    # from SEAttention import SEAttention
    # input=torch.randn((1,1,22,1000))
    # one=torch.nn.BatchNorm2d(1, False)  # 训练之前统一对数据做中心标准化，效果更好一点。
    # two=torch.nn.Conv2d(1, 10, kernel_size=1, stride=1)
    # se = SEAttention(channel=10, reduction=5)
    #
    #
    #
    # # 输出：1*40*22*47
    # output=se(two(one(input)))
    # list=[0.5243055555555556, 0.6076388888888888, 0.6041666666666666, 0.6493055555555556, 0.6180555555555556, 0.65625,
    #  0.6631944444444444, 0.6388888888888888, 0.6284722222222222, 0.6631944444444444, 0.6458333333333334,
    #  0.6701388888888888, 0.6736111111111112, 0.6701388888888888, 0.7083333333333334, 0.7013888888888888,
    #  0.6770833333333334, 0.6736111111111112, 0.6875, 0.6354166666666666, 0.6423611111111112, 0.6631944444444444,
    #  0.6944444444444444, 0.6770833333333334, 0.6423611111111112, 0.7152777777777778, 0.6701388888888888,
    #  0.6701388888888888, 0.6736111111111112, 0.6875, 0.6770833333333334, 0.6666666666666666, 0.6631944444444444,
    #  0.6736111111111112, 0.6701388888888888, 0.6527777777777778, 0.6875, 0.6423611111111112, 0.6423611111111112,
    #  0.6805555555555556, 0.6979166666666666, 0.6840277777777778, 0.6840277777777778, 0.6979166666666666,
    #  0.6458333333333334, 0.6979166666666666, 0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
    #  0.6736111111111112, 0.6631944444444444, 0.6909722222222222, 0.6805555555555556, 0.6805555555555556, 0.6875,
    #  0.7013888888888888, 0.7013888888888888, 0.65625, 0.6840277777777778, 0.6875, 0.6979166666666666,
    #  0.7048611111111112, 0.6875, 0.6909722222222222, 0.6944444444444444, 0.7048611111111112, 0.6736111111111112,
    #  0.6805555555555556, 0.6979166666666666, 0.7152777777777778, 0.6701388888888888, 0.6805555555555556,
    #  0.6701388888888888, 0.6666666666666666, 0.6805555555555556, 0.6875, 0.6736111111111112, 0.6770833333333334,
    #  0.6701388888888888, 0.6944444444444444, 0.7083333333333334, 0.6805555555555556, 0.6840277777777778,
    #  0.6944444444444444, 0.6979166666666666, 0.6701388888888888, 0.6701388888888888, 0.6875, 0.7118055555555556,
    #  0.6805555555555556, 0.7256944444444444, 0.6875, 0.6805555555555556, 0.7083333333333334, 0.6631944444444444,
    #  0.7222222222222222, 0.6631944444444444, 0.6909722222222222, 0.7083333333333334, 0.6909722222222222]
    # # list_two=[0.5729166666666666, 0.6354166666666666, 0.6006944444444444, 0.6111111111111112, 0.5972222222222222, 0.6076388888888888, 0.5972222222222222, 0.65625, 0.6354166666666666, 0.6875, 0.6840277777777778, 0.6493055555555556, 0.6666666666666666, 0.6631944444444444, 0.6388888888888888, 0.6736111111111112, 0.6527777777777778, 0.6666666666666666, 0.7118055555555556, 0.6597222222222222, 0.6875, 0.6736111111111112, 0.6597222222222222, 0.6597222222222222, 0.6840277777777778, 0.6979166666666666, 0.6597222222222222, 0.7152777777777778, 0.6597222222222222, 0.6805555555555556, 0.6666666666666666, 0.6944444444444444, 0.6666666666666666, 0.6736111111111112, 0.6909722222222222, 0.6770833333333334, 0.6840277777777778, 0.6944444444444444, 0.6805555555555556, 0.6875, 0.65625, 0.6666666666666666, 0.6805555555555556, 0.6909722222222222, 0.7013888888888888, 0.6770833333333334, 0.6770833333333334, 0.6770833333333334, 0.6701388888888888, 0.6770833333333334, 0.6701388888888888, 0.6770833333333334, 0.6701388888888888, 0.6840277777777778, 0.6805555555555556, 0.6840277777777778, 0.6909722222222222, 0.6909722222222222, 0.6909722222222222, 0.6909722222222222, 0.6875, 0.6909722222222222, 0.6701388888888888, 0.6840277777777778, 0.6805555555555556, 0.6840277777777778, 0.7013888888888888, 0.6944444444444444, 0.6909722222222222, 0.6631944444444444, 0.7013888888888888, 0.6840277777777778, 0.6631944444444444, 0.6736111111111112, 0.6979166666666666, 0.6527777777777778, 0.6875, 0.6979166666666666, 0.6666666666666666, 0.6493055555555556, 0.7013888888888888, 0.6736111111111112, 0.6875, 0.6875, 0.6631944444444444, 0.6875, 0.6979166666666666, 0.6875, 0.6770833333333334, 0.6805555555555556, 0.6840277777777778, 0.7048611111111112, 0.6875, 0.6701388888888888, 0.6458333333333334, 0.6597222222222222, 0.6944444444444444, 0.6979166666666666, 0.6979166666666666, 0.6736111111111112]0.7777777777777778, 0.4722222222222222, 0.8159722222222222, 0.6354166666666666, 0.5069444444444444, 0.5173611111111112, 0.8125, 0.7326388888888888, 0.7152777777777778]
    # list_max=max(list)
    # list=[0.69, 0.474, 0.739, 0.45, 0.43, 0.378, 0.53, 0.68, 0.69]
    # list2=[0.7118,0.5034,0.7916,0.4444,0.5555,0.4722,0.6458,0.7048,0.7118]
    # list1=[1,2,3,4,5,6,7,8,9]
    # list3 = [1, 0.5034, 0.7916, 0.4444, 0.5555, 0.4722, 0.6458, 0.7048, 0.7118]
    # function_visual(list1,list2,"1")
    # function_visual(list1,list3,"2")
    # avg=mean(lis)
    # print(torch.load("model"))

    # list_two_max=max(list_two)
    # modelone = convnext_small(pretrained=True)
    # # modeltwo = convnext_base(pretrained=True)
    # # modelthree = convnext_large(pretrained=True)
    # input_data = torch.ones((1, 3, 32, 750))
    # # output_data=model(input_data)
    # output_data1 = modelone(input_data)
    # # output_data2 = modeltwo(input_data)
    # # output_data3 = modelthree(input_data)
    # a=[0.84375, 0.6284722222222222, 0.9166666666666666, 0.6840277777777778, 0.6423611111111112, 0.5902777777777778, 0.8993055555555556, 0.8333333333333334, 0.7673611111111112]
    # b=mean(a)

    #

    # class ModelTwo(nn.Module):
    #     def __init__(self):
    #         super(ModelTwo, self).__init__()
    #         # # 输入数据：1*1*17*1600
    #         self.model1 = torch.nn.Sequential(
    #             # 1*1*22*1000输入
    #             # nn.BatchNorm2d(1, False),  # 训练之前统一对数据做中心标准化，效果更好一点。
    #             torch.nn.Conv2d(1, 40, kernel_size=(1, 20), stride=(1, 2)),  # 输出：1*40*22*478
    #             nn.BatchNorm2d(40),
    #             torch.nn.Conv2d(40, 40, kernel_size=(17, 1), stride=(1, 1)),  # 输出：1*40*1*478
    #             nn.ReLU(),
    #             torch.nn.AvgPool2d(kernel_size=(1, 45), stride=(1, 8)),  # 输出：1*40*1*55
    #             nn.BatchNorm2d(40, False),
    #             # nn.Flatten(),  # 输出：40*55
    #             # nn.Dropout(0.5),
    #             # nn.Linear(40 * 55, 4)
    #
    #         )
    #         self.model2 = torch.nn.Sequential(
    #             # 1*1*22*1000输入
    #             # nn.BatchNorm2d(1, False),  # 训练之前统一对数据做中心标准化，效果更好一点。
    #             torch.nn.Conv2d(1, 40, kernel_size=(1, 40), stride=(1, 2)),  # 输出：1*40*22*478
    #             nn.ZeroPad2d((5, 5, 0, 0)),
    #             nn.BatchNorm2d(40),
    #             torch.nn.Conv2d(40, 40, kernel_size=(17, 1), stride=(1, 1)),  # 输出：1*40*1*478
    #             nn.ReLU(),
    #             torch.nn.AvgPool2d(kernel_size=(1, 45), stride=(1, 8)),  # 输出：1*40*1*55
    #             nn.BatchNorm2d(40, False),
    #             # nn.Flatten(),  # 输出：40*55
    #             # nn.Dropout(0.5),
    #             # nn.Linear(40 * 55, 4)
    #
    #         )
    #         self.model3 = torch.nn.Sequential(
    #             # 1*1*22*1000输入
    #             # nn.BatchNorm2d(1, False),  # 训练之前统一对数据做中心标准化，效果更好一点。
    #             torch.nn.Conv2d(1, 40, kernel_size=(1, 60), stride=(1, 2)),  # 输出：1*40*22*478
    #             nn.ZeroPad2d((10, 10, 0, 0)),
    #             nn.BatchNorm2d(40),
    #             torch.nn.Conv2d(40, 40, kernel_size=(17, 1), stride=(1, 1)),  # 输出：1*40*1*478
    #             nn.ReLU(),
    #             torch.nn.AvgPool2d(kernel_size=(1, 45), stride=(1, 8)),  # 输出：1*40*1*55
    #             nn.BatchNorm2d(40, False),
    #             # nn.Flatten(),  # 输出：40*55
    #             # nn.Dropout(0.5),
    #             # nn.Linear(40 * 55, 4)
    #
    #         )
    #
    #         self.block = torch.nn.Sequential(
    #             torch.nn.AvgPool2d((1, 4)),
    #             nn.Flatten(),  # 输出：40*55
    #             nn.Dropout(0.5),
    #             nn.Linear(2760, 2)
    #
    #         )
    #
    #     def forward(self, x):
    #         out1 = self.model1(x)
    #         out2 = self.model2(x)
    #         out3 = self.model3(x)
    #         out4 = torch.cat([out1, out2, out3], dim=1)
    #         out5 = self.block(out4)
    #         # 试一下这个softmax
    #         return nn.functional.log_softmax(out5)
    #     # inception_two
    #
    # model=ModelTwo()

    a=torch.rand((10,4,4))

    b=nn.BatchNorm1d(4)

    c=b(a)

    print("Ok")



def function_trainModels():
    # 模型的选择:1/2/3/4_改进（lr缩小，最后的dropout去掉）
    model_num = [1]
    accuracy_all = []
    for model_i in model_num:
        result=function_three(model_i)
        accuracy_all.append(result)
        print("------------第{}个实验模型 的运行结果:{}--------------".format(model_i,result))



    print(accuracy_all)
    print("-------------------------end----------------------------------------")


if __name__=='__main__':
   function_trainModels()