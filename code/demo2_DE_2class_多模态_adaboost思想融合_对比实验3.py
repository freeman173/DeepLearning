
"""

对比实验：
    将21个样本个体单独做一次数据测试实验：
        每个个体做5折交叉实验并计算正确率







"""
from sklearn.model_selection import StratifiedKFold
from timm.models.convnext import convnext_small
from torch.optim import lr_scheduler
import torch
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
    model_eeg =IEModelSevenEeg()
    model_eeg.load_state_dict(canshu_eeg)

    canshu_eog = torch.load(pthName2 + ".pth")
    model_eog = IEModelSevenEog()
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





# 读取眼电与脑电数据(OK)
def function_readData(index):
    import torch
    import numpy as np
    import os
    from scipy.io import loadmat
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    RawdataNames = ['1_20151124_noon_2.npy', '2_20151106_noon.npy', '3_20151024_noon.npy', '4_20151105_noon.npy',
                    '5_20141108_noon.npy', '6_20151121_noon.npy', '7_20151015_night.npy', '8_20151022_noon.npy',
                    '9_20151017_night.npy', '10_20151125_noon.npy', '11_20151024_night.npy', '12_20150928_noon.npy',
                    '13_20150929_noon.npy', '14_20151014_night.npy', '15_20151126_night.npy', '16_20151128_night.npy',
                    '17_20150925_noon.npy', '18_20150926_noon.npy', '19_20151114_noon.npy', '20_20151129_night.npy',
                    '21_20151016_noon.npy']


    # 读取测试数据
    eegData1=np.load('Data/dataset_DE/'+RawdataNames[index])
    eegData2 = torch.tensor(eegData1)
    eegData3=eegData2.view(885, 16, 17, 5)
    eegDataTensor = torch.transpose(eegData3, 1, 2)

    eogDataTensor = torch.tensor(np.load('SEED-VIG/EOG_Feature/3features_npy/' + RawdataNames[index]))

    LabelData = np.load('Data/label_npy/' + RawdataNames[index]).tolist()
    Label = []
    for i in LabelData:
        if i[0] < 0.35:
            Label.append(0)
        else:
            Label.append(1)

    LabelTensor = torch.tensor(Label)


    return eegDataTensor,eogDataTensor,LabelTensor






#多模态融合训练
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

    #模型超参数
    Batch_Size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for index in range(1,21):
        start = time.time()
        # if index==1:
        #     break
        print("----------这是第{}个被试的交叉验证----------".format(index+1))


        # 读取数据：训练集、测试集
        eegDataTensor, eogDataTensor, LabelTensor=function_readData(index)



        EEGDataNumpy = eegDataTensor.numpy()
        EOGDataNumpy = eogDataTensor.numpy()
        LabelNumpy = LabelTensor.numpy()

        np.random.seed(30)
        np.random.shuffle(EEGDataNumpy)
        np.random.seed(30)
        np.random.shuffle(EOGDataNumpy)
        np.random.seed(30)
        np.random.shuffle(LabelNumpy)


        #读取数据
        eegDataTensor= torch.from_numpy(EEGDataNumpy).contiguous().view(885,17, -1)
        eogDataTensor = torch.from_numpy(EOGDataNumpy).contiguous().view(885,3, -1)
        labelTensor = torch.from_numpy(LabelNumpy)

        print("_______")




        # 一共循环5次：从第一份开始，每份轮流做验证集
        number_cross1 = 0
        # 用于记录每次交叉验证中测试集上的正确率（从accuracy_list中选取最大的）
        accuracyMaxLists = []

        # 分层5折交叉:因为每次的数据的顺序都一样，所以5折划分之后，每次也都一样
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index1, test_index1 in skf.split(eegDataTensor, labelTensor):
            number_cross1 += 1
            if(number_cross1!=1):#一个个体只跑一次交叉，后面正常实验时，把这里注释掉就好
                break;
            number_cross2 = 0
            for train_index2, test_index2 in skf.split(eogDataTensor, labelTensor):
                number_cross2 += 1
                if (number_cross2!= 1):#一个个体只跑一次交叉，后面正常实验时，把这里注释掉就好
                    break;
                if number_cross1 == number_cross2:
                    # 用于测试
                    # print(labelTensor[test_index1])
                    # print(labelTensor[test_index2])


                    print("----------这是第{}交叉验证----------".format(number_cross2))

                    # print("Train:", train_index, "Validation:", test_index)
                    # 拆分数据:训练集、验证集、测试集
                    eegDataTrain,eegDataVerify, eegDataTest = eegDataTensor[train_index1][177:], eegDataTensor[train_index1][:177],eegDataTensor[test_index1]
                    eogDataTrain, eogDataVerify,eogDataTest = eogDataTensor[train_index2][177:],eogDataTensor[train_index2][:177], eogDataTensor[test_index2]
                    labelTrain,labelVerify, labelTest = labelTensor[train_index2][177:], labelTensor[train_index2][:177], labelTensor[test_index2]



                    # 继续封装数据
                    # 12213 % 32=381
                    train_loader = DataLoader(dataset=TensorDataset(eegDataTrain, eogDataTrain, labelTrain),
                                              batch_size=Batch_Size, shuffle=True, drop_last=True)
                    train_loader_count=16
                    # 4071 % 32=127
                    verify_loader = DataLoader(dataset=TensorDataset(eegDataVerify, eogDataVerify, labelVerify),
                                              batch_size=Batch_Size, shuffle=True, drop_last=True)
                    verify_loader_count = 5
                    # 4071 % 32 =127
                    test_loader = DataLoader(dataset=TensorDataset(eegDataTest,eogDataTest, labelTest),
                                             batch_size=Batch_Size, shuffle=True, drop_last=True)
                    test_loader_count = 5




                    if model_i == 1:
                        modelEEG = IEModelSevenEeg().to(device)
                        modelEOG = IEModelSevenEog().to(device)
                    elif model_i == 2:
                        modelEEG = ESTCNN().to(device)
                        modelEOG = ESTCNNTEOG().to(device)
                    elif model_i == 3:
                        modelEEG = AMCNN_DGCN().to(device)
                        modelEOG = AMCNN_DGCN_EOG().to(device)
                    elif model_i == 4:
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
                    EPOCH = 150
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
                        for eeg, eog, targets in train_loader:
                            outputEEG ,JC= modelEEG(eeg.to(torch.float32).to(device))
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
                            for eeg_test, eog_test, targets_test in verify_loader:

                                outputsEEG_test,JC = modelEEG(eeg_test.to(torch.float32).to(device))
                                outputsEOG_test = modelEOG(eog_test.to(torch.float32).to(device))

                                # 验证集上的损失值暂时用不到，就不做进一步开发了！
                                # loss_test_eeg = loss_fn_EEG(outputsEEG_test, targets_test.to(torch.long).to(device))
                                # loss_test_eog = loss_fn_EOG(outputsEOG_test, targets_test.to(torch.long).to(device))

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
                    print(accuracyEEG_list)
                    print("\n")
                    print(accuracyEOG_list)





                    #取出两个子模型的最大预测值做一个权重分配:
                    print("第{}次交叉验证集上的最大正确率:eeg:{}||eog:{}".format(number_cross2,max(accuracyEEG_list),max(accuracyEOG_list) ))
                    X=max(accuracyEEG_list);Y=max(accuracyEOG_list)
                    if X==1:#正确率为1，改小一点，不然权重不好分配
                        X=0.98
                    if Y==1:
                        Y=0.98
                    eegModelWeight ,eogModelWeight = computeWeight(X,Y)
                    print("第{}次交叉测试集上的权重分配:eeg:{}||eog:{}".format(number_cross2, eegModelWeight,eogModelWeight))



                    # 保存训练之后的模型
                    saveModel(modelEEG,"eeg_"+str(number_cross1))
                    saveModel(modelEOG,"eog_"+str(number_cross1))

                  #读取模型用来测试
                    modelEEG_test,modelEOG_test=loadModel("eeg_"+str(number_cross1),"eog_"+str(number_cross1))
                    modelEEG_test=modelEEG_test.to(device)
                    modelEOG_test=modelEOG_test.to(device)



                # 把所有数据放在一起，便于统一存取
                    data_eeg = {"accuracy": [], "precision": [], "recall": [], "f1": []}
                    data_eog = {"accuracy": [], "precision": [], "recall": [], "f1": []}
                    data_ronghe = {"accuracy": [], "precision": [], "recall": [], "f1": []}



                #
                # # 在测试集上测试10次，然后取融合最大值这一组
                    JC_all = torch.zeros(0, 17, 5).to(device)
                    label_all=torch.zeros(0).to(device)
                    for count_test in range(1,101):
                        # 准备数据
                        TNEEG = 0;TPEEG = 0;FPEEG = 0;FNEEG = 0
                        TNEOG = 0;TPEOG = 0;FPEOG = 0;FNEOG = 0;
                        TN = 0;TP = 0;FP = 0;FN = 0
                        zes = torch.zeros(32).to(device);ons = torch.ones(32).to(device)
                        with torch.no_grad():  # 将模型的梯度清零，进行测试
                            # 测试集上测试数据
                            # 验证集上验证数据

                            for eeg_test, eog_test, targets_test in test_loader:

                                # 预测的结果
                                outputsEEG_test,JC = modelEEG_test(
                                    eeg_test.to(torch.float32).to(device))
                                outputsEOG_test = modelEOG_test(eog_test.to(torch.float32).to(device))
                                if count_test==100:
                                    JC_all = torch.cat([JC_all,JC], dim=0)
                                    label_all = torch.cat([label_all,targets_test.to(device)],dim=0)
                                # 将结果加权融合了一手
                                blending_y_pred = outputsEEG_test * eegModelWeight+ outputsEOG_test * eogModelWeight

                                pred_eeg = outputsEEG_test.argmax(1)       # eeg预测结果
                                pred_eog = outputsEOG_test.argmax(1)       #eog 预测结果
                                pred_ronghe = blending_y_pred.argmax(1)    # 融合预测结果
                                label = targets_test.to(device)  # 真实标签


                                # 计算TP/TN/FP/FN
                                TPEEG += ((pred_eeg == ons) & (label == ons)).sum()
                                TNEEG += ((pred_eeg == zes) & (label == zes)).sum()
                                FPEEG += ((pred_eeg == ons) & (label == zes)).sum()
                                FNEEG += ((pred_eeg == zes) & (label == ons)).sum()

                                TPEOG += ((pred_eog == ons) & (label == ons)).sum()
                                TNEOG += ((pred_eog == zes) & (label == zes)).sum()
                                FPEOG += ((pred_eog == ons) & (label == zes)).sum()
                                FNEOG += ((pred_eog == zes) & (label == ons)).sum()

                                TP += ((pred_ronghe == ons) & (label == ons)).sum()
                                TN += ((pred_ronghe == zes) & (label == zes)).sum()
                                FP += ((pred_ronghe == ons) & (label == zes)).sum()
                                FN += ((pred_ronghe == zes) & (label == ons)).sum()


                            # 计算各项指标
                            # 准确率
                            accuracyeeg=((TPEEG + TNEEG) / (TPEEG + TNEEG + FPEEG + FNEEG)).item()
                            accuracyeog =((TPEOG + TNEOG) / (TPEOG + TNEOG + FPEOG + FNEOG)).item()
                            accuracy =((TP + TN) / (TP + TN + FP + FN)).item()
                            data_eeg["accuracy"].append (accuracyeeg)
                            data_eog["accuracy"].append(accuracyeog)
                            data_ronghe["accuracy"].append(accuracy)


                            # 精确率（查准率）:即正确预测为正的占全部预测为正的比例
                            precisioneeg =(TPEEG / (TPEEG + FPEEG)).item()
                            precisioneog =(TPEOG / (TPEOG + FPEOG)).item()
                            precision =(TP / (TP + FP)).item()
                            data_eeg["precision"].append (precisioneeg)
                            data_eog["precision"].append (precisioneog)
                            data_ronghe["precision"].append (precision)


                            # 召回率（查全率）:即正确预测为正的占全部实际为正的比例
                            recalleeg =(TPEEG / (TPEEG + FNEEG)).item()
                            recalleog =(TPEOG / (TPEOG + FNEOG)).item()
                            recall =(TP / (TP + FN)).item()
                            data_eeg["recall"].append(recalleeg)
                            data_eog["recall"].append(recalleog)
                            data_ronghe["recall"].append(recall)

                            # F1值：越大越好
                            f1eeg =(2 * TPEEG / (2 * TPEEG + FPEEG + FNEEG)).item()
                            f1eog =(2 * TPEOG / (2 * TPEOG + FPEOG + FNEOG)).item()
                            f1=(2 * TP / (2 * TP + FP + FN)).item()
                            data_eeg["f1"].append(f1eeg)
                            data_eog["f1"].append(f1eog)
                            data_ronghe["f1"].append(f1)

                            print("第{}次交叉验证第{}次整体EEG测试集上的正确率:{}||精确率:{}||召回率:{}||f1:{}".format(number_cross2,count_test,accuracyeeg,precisioneeg,
                                                                                               recalleeg,f1eeg))

                            print("第{}次交叉验证第{}次整体EOG测试集上的正确率:{}||精确率:{}||召回率:{}||f1:{}".format(number_cross2, count_test,
                                                                                               accuracyeog, precisioneog,
                                                                                               recalleog, f1eog))
                            print("第{}次交叉验证第{}次整体融合测试集上的正确率:{}||精确率:{}||召回率:{}||f1:{}".format(number_cross2, count_test,
                                                                                               accuracy, precision,
                                                                                               recall, f1))

                    from KeJieShi import function_visual_topomap
                    print(JC_all.tolist()[:20])
                    print(label_all.tolist()[:20])
                    function_visual_topomap(JC_all.tolist(),label_all.tolist(),index+1)
                    print("————————————————————————————————————————————————————————————————————————————————————————————————————————————————")





        #             #把结果融合最大准确率值的下标得到
        #             index = data_ronghe['accuracy'].index(max(data_ronghe["accuracy"]))
        #             index_eeg=data_eeg['accuracy'].index(max(data_eeg["accuracy"]))
        #             index_eog = data_eog['accuracy'].index(max(data_eog["accuracy"]))
        #
        #
        #             # 把下标对应的每组值得到并打印
        #             print("第{}次交叉验证整体eeg测试集上的正确率:{}||精确率:{}||召回率:{}||f1:{}".format(number_cross2,  data_eeg['accuracy'][index_eeg],
        #             data_eeg['precision'][index],
        #             data_eeg['recall'][index],
        #             data_eeg['f1'][index]))
        #
        #             print("第{}次交叉验证整体eog测试集上的正确率:{}||精确率:{}||召回率:{}||f1:{}".format(number_cross2, data_eog['accuracy'][index_eog],
        #                                                                          data_eog['precision'][index],
        #                                                                          data_eog['recall'][index],
        #                                                                          data_eog['f1'][index]))
        #
        #             print("第{}次交叉验证整体融合测试集上的正确率:{}||精确率:{}||召回率:{}||f1:{}".format(number_cross2, data_ronghe['accuracy'][index],
        #                                                                          data_ronghe['precision'][index],
        #                                                                          data_ronghe['recall'][index],
        #                                                                          data_ronghe['f1'][index]))
        #
        #
        #             # 把每次交叉验证的融合最大准确率进行记录
        #             accuracyMaxLists.append(data_ronghe['accuracy'][index])
        #
        #
        #
        #
        #             # 将数据写入文件
        #             # file_handle = open("Result/29.txt", mode='a+')
        #             # file_handle.write(str(data_eeg['accuracy'][index])+","+str(data_eeg['precision'][index])+","+str(data_eeg['recall'][index])+","+str(data_eeg['f1'][index])+"\n")
        #             #
        #             # file_handle.write(
        #             #     str(data_eog['accuracy'][index]) + "," + str(data_eog['precision'][index]) + "," + str(
        #             #         data_eog['recall'][index]) + "," + str(data_eog['f1'][index]) + "\n")
        #             #
        #             # file_handle.write(
        #             #     str(data_ronghe['accuracy'][index]) + "," + str(data_ronghe['precision'][index]) + "," + str(
        #             #         data_ronghe['recall'][index]) + "," + str(data_ronghe['f1'][index]) + "\n")
        #             # file_handle.close()
        #
        #
        #             #本次交叉验证结束咯
        #             now = time.time()
        #             print("本次交叉训练花费时间：{}s".format(now - start))
        #
        # # 计算5次交叉验证的正确率均值并写入文件
        # result1 = np.mean(accuracyMaxLists)
        # print(str(accuracyMaxLists) + "||" + str(result1) + "\n")
        # # file_handle = open("Result/34.txt", mode='a+')
        # # file_handle.write(str(accuracyMaxLists)+"||"+str(result1)+"\n")
        # # file_handle.close()



    return 0








#选择模型进行训练
def function_trainModels():
    # 模型的选择:1/2/3/4_改进（lr缩小，最后的dropout去掉）
    model_num = [1]
    #计算结果的记录
    # accuracy_all = []
    for model_i in model_num:
            function_three(model_i)


    # print(accuracy_all)
    print("-------------------------end----------------------------------------")







if __name__=='__main__':
    # function_three(2)
    # function_test()
    function_trainModels()
    # function_readData()









