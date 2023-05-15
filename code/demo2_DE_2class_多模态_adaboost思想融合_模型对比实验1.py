
"""

将数据集按照4:1划分为训练集与验证集：
        测试4个模型在eeg、eog： accuracy随着epoch的表现




"""
from sklearn.model_selection import StratifiedKFold
from timm.models.convnext import convnext_small
from torch import nn
from torch.optim import lr_scheduler
import torch
from torchvision.transforms import transforms

from CNN_LSTM import CNN_LSTM,CNN_LSTM_EOG,CNN_LSTM_Two,CNN_LSTM_EOG_Two
# from IENET_Model_Seven_3_eeg_eog_DE_jieguoronghe import IEModelSevenEeg,IEModelSevenEog
from IENET_Model_Seven_3_eeg_eog_DE_jieguoronghe_2 import IEModelSevenEeg,IEModelSevenEog,IEModelSevenEog_xiaorong,IEModelSevenEeg_xiaorong,IEModelSevenEog_xiaorong_3,IEModelSevenEeg_xiaorong_3,IEModelSevenEog_xiaorong_9,IEModelSevenEeg_xiaorong_9
from AMCNN_DGCN import AMCNN_DGCN,AMCNN_DGCN_EOG
from ETSCNN import ESTCNN,ESTCNNTEOG

#可视化函数
def function_visual(x, y, title):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.savefig("{}.png".format(title))
    plt.show()


#保存模型
def saveModel(model,name):
    # 保存模型参数
    torch.save(model.state_dict(), name+".pth")
    #创建模型
    # model = Model()
    #加载参数并传入模型
    # canshu = torch.load("model.pth")
    # model.load_state_dict(canshu)


# 读取模型（当模型发生变化时，每次都要手动过来改，麻烦的很，暂时先这样问题不大）
def loadModel(pthName1,pthName2):

    #加载参数并传入模型

    canshu_eeg = torch.load(pthName1+".pth")
    model_eeg = ESTCNN()
    model_eeg.load_state_dict(canshu_eeg)

    canshu_eog = torch.load(pthName2 + ".pth")
    model_eog = ESTCNNTEOG()
    model_eog.load_state_dict(canshu_eog)


    return model_eeg.eval(),model_eog.eval()


#计算模型权重
def computeWeight(accuracyEEG,accuracyEOG):
    import numpy as np
    a = np.log(accuracyEEG / (1 - accuracyEEG))
    b = np.log(accuracyEOG / (1 - accuracyEOG))
    result = a + b
    eegModelWeight = a / result
    eogModelWeight = b / result
    return eegModelWeight,eogModelWeight



#用作测试的函数
def function_readData_test():
    import torch
    import numpy as np
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    #读取训练数据
    #17*16*5:17个脑电频道，将1600个点按100切分为16组，在每一组里面提取5个微分商
    # TrainDataTensor =torch.transpose(torch.tensor(np.load('Data/dataset_DE/train_set.3features_npy')), 1, 2)
    # TrainLabelData = np.load('Data/dataset_DE/train_label.3features_npy').tolist()
    # TrainLabel = []
    # for i in TrainLabelData:
    #     if i[0] < 0.35:
    #         TrainLabel.append(0)
    #     else:
    #         TrainLabel.append(1)
    # TrainLabelTensor = torch.tensor(TrainLabel)
    #
    # #读取测试数据
    TestDataTensor1 = torch.transpose(torch.tensor(np.load('Data/dataset_DE/data_DE.npy')), 1, 2)
    TrainLabelData = np.load('Data/dataset_DE/labels_all.npy').tolist()
    TestLabel = []
    # for i in TestLabelData:
    #     if i[0] < 0.35:
    #         TestLabel.append(0)
    #     else:
    #         TestLabel.append(1)

    LabelTensor = torch.tensor(TestLabel)
    # #
    # DataTensor = torch.cat([TrainDataTensor, TestDataTensor], dim=0)
    # LabelTensor1 = torch.cat([TrainLabelTensor, TestLabelTensor], dim=0)




    # label2 = np.load('Data/label_npy/label_all.3features_npy')
    # TrainLabelData = label.tolist()
    # TrainLabel = []
    # for i in TrainLabelData:
    #     if i[0] < 0.35:
    #         TrainLabel.append(0)
    #     else:
    #         TrainLabel.append(1)
    # LabelTensor2 = torch.tensor(TrainLabel)

    return 1


# 向眼电数据添加高斯噪声
def function_addNoise_1(y):
    import random

    import numpy as np
    from matplotlib import pyplot as plt

    def gauss_noisy(x, y):
        """
        对输入数据加入高斯噪声
        :param x: x轴数据
        :param y: y轴数据
        :return:
        """
        mu = 0

        sigma = 0.2  # 0.0就是原数据
        for i in range(len(x)):
            pass
            # x[i] += random.gauss(mu, sigma)
            y[i] += random.gauss(mu, sigma)

        print("________")

    # 在0-5的区间上生成50个点作为测试数据
    x = np.linspace(0, 10, 36, endpoint=True)
    y = y.numpy()
    # 加入高斯噪声
    gauss_noisy(x, y)

    # 画出这些点
    plt.plot(x, y, linestyle='', marker='.')

    plt.show()

    print("________________")


def gauss_noisy(data_raw, noise_i):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    import random
    mu = 0
    sigma = noise_i  # 0.0就是原数据

    for i in range(data_raw.shape[0]):
        for j in range(data_raw.shape[1]):
            for K in range(data_raw.shape[2]):
                data_raw[i][j][K] += random.gauss(mu, sigma)
            # print("______")
    return data_raw


def function_addNoise(EOGDataTensor, noise_i):
    # 读取数据并打乱

    EEGDataTensor, EOGDataTensor, LabelTensor = function_readData()

    #
    jc = nn.BatchNorm1d(3)
    EOGDataNumpy = jc(EOGDataTensor.to(torch.float32)).detach().numpy()

    data_guiyihua_noise = gauss_noisy(EOGDataNumpy, noise_i)

    return data_guiyihua_noise



# 读取眼电与脑电数据(OK)
def function_readData():
    import torch
    import numpy as np
    import os
    from scipy.io import loadmat
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 读取眼电
    eogData = np.load('SEED-VIG/EOG_Feature/3features_npy/eog_all.npy')
    EogTrainDataTensor = torch.tensor(eogData)


    #读取脑电
    EegTrainDataTensor = torch.transpose(torch.tensor(np.load('Data/dataset_DE/data_DE.npy')), 1, 2)



    #读取标签
    label = np.load('Data/label_npy/label_all.npy')
    TrainLabelData =label.tolist()
    TrainLabel = []
    for i in TrainLabelData:
        if i[0] < 0.35:
            TrainLabel.append(0)
        else:
            TrainLabel.append(1)
    TrainLabelTensor = torch.tensor(TrainLabel)








    return EegTrainDataTensor,EogTrainDataTensor,TrainLabelTensor




#多模态融合训练
def function_three(model_i,noise_i):
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

    #模型超参数
    Batch_Size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #读取数据并打乱

    EEGDataTensor, EOGDataTensor, LabelTensor = function_readData()


    EEGDataNumpy = EEGDataTensor.numpy()
    EOGDataNumpy =function_addNoise(EOGDataTensor,noise_i)
    LabelNumpy = LabelTensor.numpy()


    np.random.seed(30)
    np.random.shuffle(EEGDataNumpy)
    np.random.seed(30)
    np.random.shuffle(EOGDataNumpy)
    np.random.seed(30)
    np.random.shuffle(LabelNumpy)


    #读取数据
    eegDataTensor= torch.from_numpy(EEGDataNumpy).contiguous().view(20355,17, -1)
    eogDataTensor = torch.from_numpy(EOGDataNumpy).contiguous().view(20355,3, -1)
    labelTensor = torch.from_numpy(LabelNumpy)



    eegDataTrain,eegDataVerify= eegDataTensor[:16284], eegDataTensor[16284:]
    eogDataTrain, eogDataVerify = eogDataTensor[:16284],eogDataTensor[16284:],
    labelTrain,labelVerify = labelTensor[:16284], labelTensor[16284:]



                # 继续封装数据
                # 12213 % 32=508
    train_loader = DataLoader(dataset=TensorDataset(eegDataTrain, eogDataTrain, labelTrain),
                                          batch_size=Batch_Size, shuffle=True, drop_last=True)

    train_loader_count=508
                # 4071 % 32=127
    verify_loader = DataLoader(dataset=TensorDataset(eegDataVerify, eogDataVerify, labelVerify),
                                          batch_size=Batch_Size, shuffle=True, drop_last=True)

    verify_loader_count=127



    if model_i == 1:
        modelEEG =IEModelSevenEeg().to(device)
        modelEOG =IEModelSevenEog().to(device)
    elif model_i==2:
        modelEEG = ESTCNN().to(device)
        modelEOG = ESTCNNTEOG().to(device)
    elif model_i==3:
        modelEEG = AMCNN_DGCN().to(device)
        modelEOG = AMCNN_DGCN_EOG().to(device)
    elif model_i==4:
        modelEEG = CNN_LSTM_Two().to(device)
        modelEOG = CNN_LSTM_EOG_Two().to(device)


                # 给定优化器:加了L2正则化;momentum,在SGD的基础上加一个动量，如果当前收敛效果好，就可以加速收敛，如果不好，则减慢它的步伐。
    optimizerEEG = optim.SGD(modelEEG.parameters(), lr=0.01, weight_decay=0.01, momentum=0.3)
    optimizerEOG = optim.SGD(modelEOG.parameters(), lr=0.01, weight_decay=0.01, momentum=0.3)

                # # 定义学习率与轮数关系的函数
    lambda1 = lambda epoch: 0.99 ** epoch  # 学习率 = 0.95**(轮数)
    schedulerEEG = lr_scheduler.LambdaLR(optimizerEEG, lr_lambda=lambda1)
    schedulerEOG = lr_scheduler.LambdaLR(optimizerEOG, lr_lambda=lambda1)



                # optimizer_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
                # 损失函数
    loss_fn_EEG = torch.nn.CrossEntropyLoss()
    loss_fn_EOG = torch.nn.CrossEntropyLoss()


                # 初始化 early_stopping 对象
    patience = 50  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stoppingEEG = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    early_stoppingEOG = EarlyStopping(patience, verbose=True)
                # 相关训练数据
    EPOCH = 100
    epoch_list = []

                # eeg：训练集损失值、验证集损失值、验证集正确率     ||eog：训练集损失值、验证集损失值、验证集正确率
    lossTrainEEG_list = []
    lossTestEEG_list = []
    accuracyEEG_list = []

    lossTrainEOG_list = []
    lossTestEOG_list = []
    accuracyEOG_list = []



                # 一个计数器:只输出最后一次训练集的数据
    count_train = 0
    for epoch in range(EPOCH):
        epoch_list.append(epoch + 1)
        loss_train_total_eeg = 0  # 每轮训练的损失值初始化为0
        loss_train_total_eog = 0  # 每轮训练的损失值初始化为0
        # JC_all=torch.zeros(0,17,5).to(device)
        for eeg, eog, targets in train_loader:

            outputEEG = modelEEG(eeg.to(torch.float32).to(device))
            # JC_all=torch.cat([JC_all,JC],dim=0)
            outputE0G = modelEOG(eog.to(torch.float32).to(device))


            loss_train_eeg = loss_fn_EEG(outputEEG, targets.to(torch.long).to(device))
            loss_train_eog = loss_fn_EOG(outputE0G, targets.to(torch.long).to(device))

            optimizerEEG.zero_grad()  # 优化之前必备工作：梯度清零
            optimizerEOG.zero_grad()

            loss_train_eeg.backward()  # 后向传播求导
            loss_train_eog.backward()

            optimizerEEG.step()  # 优化参数
            optimizerEOG.step()

                        # 将损失值的总和累加到一块
            loss_train_total_eeg += loss_train_eeg.item()
            loss_train_total_eog += loss_train_eog.item()


            count_train += 1
            if count_train % train_loader_count == 0:
                            # 计算每轮的平均损失值
                loss_train_avg_eeg = loss_train_total_eeg / train_loader_count
                loss_train_avg_eog = loss_train_total_eog / train_loader_count

                print("第{}次训练的EEG损失值(训练集)：{}".format(epoch + 1, loss_train_avg_eeg))
                lossTrainEEG_list.append(loss_train_avg_eeg)

                print("第{}次训练的EOG损失值(训练集)：{}".format(epoch + 1, loss_train_avg_eog))
                lossTrainEOG_list.append(loss_train_avg_eog)
                count_train = 0




        with torch.no_grad():  # 将模型的梯度清零，进行验证
            numberEEG_right=0
            numberEOG_right=0

                        # 验证集上验证数据
            JC_all = torch.zeros(0, 17, 5).to(device)
            for eeg_test, eog_test, targets_test in verify_loader:

                outputsEEG_test = modelEEG(eeg_test.to(torch.float32).to(device))
                outputsEOG_test = modelEOG(eog_test.to(torch.float32).to(device))
                # if epoch==199:
                #     JC_all = torch.cat([JC_all, JC], dim=0)

                            # 统计每次测试集合中，正确的个数
                numberEEG_right += (outputsEEG_test.argmax(1) == targets_test.to(device)).sum().item()
                numberEOG_right += (outputsEOG_test.argmax(1) == targets_test.to(device)).sum().item()



                        #计算验证集上的正确率
            accuracyEEG = numberEEG_right / (verify_loader_count*Batch_Size)
            accuracyEOG = numberEOG_right / (verify_loader_count*Batch_Size)


            print("第{}次整体验证集上的正确率:eeg:{}||eog:{}".format(epoch + 1, accuracyEEG,accuracyEOG))
            accuracyEEG_list.append(accuracyEEG)
            accuracyEOG_list.append(accuracyEOG)


                    # 学习率衰减
            schedulerEEG.step()
            schedulerEOG.step()
            print("epoch={}, eeg_lr={}".format(epoch, optimizerEEG.state_dict()['param_groups'][0]['lr']))
            print("epoch={}, eog_lr={}".format(epoch, optimizerEOG.state_dict()['param_groups'][0]['lr']))
                    # 画图时让x与y有相同维度

            early_stoppingEEG(loss_train_eeg, modelEEG)
            early_stoppingEOG(loss_train_eog,modelEOG)

            if early_stoppingEEG.early_stop or early_stoppingEOG.early_stop:
                break;




    # 将训练集的损失值与验证集的正确率情况绘制一手
    print("第{}模型的实验数据".format(model_i))
    print(accuracyEEG_list)
    print(max(accuracyEEG_list))
    print("\n")
    print(accuracyEOG_list)
    print(max(accuracyEOG_list))

    file_handle = open("Result/35.txt", mode='a+')
    file_handle.write(str(max(accuracyEOG_list)) +"\n")
    file_handle.close()









#选择模型进行训练
def function_trainModels():
    # 模型的选择:1/2/3/4_改进（lr缩小，最后的dropout去掉）
    model_num = [1,2,3,4]
    noise_num=[0.05,0.10,0.15,0.20]
    #计算结果的记录
    # accuracy_all = []
    for model_i in model_num:
        for noise_i in noise_num:
            function_three(model_i,noise_i)


    # print(accuracy_all)
    print("-------------------------end----------------------------------------")








#向眼电数据添加高斯噪声
def function_addNoise_1_jc(y,noise_i):

    import random

    import numpy as np
    from matplotlib import pyplot as plt


    def gauss_noisy(y,noise_i):
        """
        对输入数据加入高斯噪声
        :param x: x轴数据
        :param y: y轴数据
        :return:
        """
        mu = 0.0

        sigma = noise_i#0.0就是原数据
        for i in range(len(y)):
            pass
            # x[i] += random.gauss(mu, sigma)
            y[i] += random.gauss(mu, sigma)

        print("________")



    # 在0-5的区间上生成50个点作为测试数据
    x = np.linspace(0, 10, 36, endpoint=True)
    # 加入高斯噪声
    gauss_noisy(y,noise_i)

    # 画出这些点
    plt.plot(x, y, linestyle='', marker='.')


    plt.show()





    print("________________")









def function_addNoise_jc():
    # 读取数据并打乱

    EEGDataTensor, EOGDataTensor, LabelTensor = function_readData()

    #
    jc=nn.BatchNorm1d(3)
    EOGDataNumpy = jc(EOGDataTensor.to(torch.float32)).detach().numpy()
    # EOGDataNumpy = EOGDataTensor.to(torch.float32).detach().numpy()

    noise_i=0.08
    data_guiyihua_noise=function_addNoise_1_jc(EOGDataNumpy[100][1],noise_i)


    return data_guiyihua_noise








if __name__=='__main__':

    function_trainModels()
    # function_addNoise_jc()









