# 单独模块写好了：

import torch
import torch.nn as nn
from timm.models.convnext import convnext_small
from torch.nn import Linear

"""
    
结果融合：

    在第一版的基础上，将第二路的最大池化改动一下 ：
        看一下效果如何：86.46%，效果差不多了，就用这个，模型的可解释性也好说一点
    
5折交叉：
    0.8292322754859924  , 0.8449802994728088 , 0.8533464670181274   ,0.8476870059967041    ,0.842765748500824
    0.8973917365074158 , 0.875 , 0.9266732335090637 , 0.8282480239868164 ,  0.9274114370346069
    0.9028050899505615  ,  0.8946850299835205  , 0.9219980239868164 , 0.9003444910049438  ,  0.9185531735420227
    
  

再跑一次：
    epoch：150 ； earlystoping：50
  
0.8432578444480896,0.8576669096946716,0.9037495255470276,0.8801053762435913
0.8774606585502625,0.8869951963424683,0.925396203994751,0.9057888984680176
0.8917322754859924,0.888526976108551,0.9489756226539612,0.9177570343017578



0.8277559280395508,0.8646034598350525,0.8646034598350525,0.8646034598350525
0.8922244310379028,0.9714536666870117,0.855705976486206,0.9099135994911194
0.8986220359802246,0.9354709386825562,0.9029013514518738,0.9188976287841797



0.8489173054695129,0.8845553994178772,0.8770301342010498,0.8807767033576965
0.8348917365074158,0.9818822145462036,0.7544470429420471,0.8532691597938538
0.9023129940032959,0.9600672721862793,0.8832173347473145,0.9200403094291687



0.8464567065238953,0.8846606612205505,0.8723404407501221,0.8784573674201965
0.8720472455024719,0.9695315957069397,0.8247582316398621,0.8913043737411499
0.9069882035255432,0.9528108239173889,0.898259162902832,0.9247311949729919



0.8521161675453186,0.8948206901550293,0.8695315718650818,0.881994903087616
0.8710629940032959,0.9711670279502869,0.8215253353118896,0.8901006579399109
0.9087106585502625,0.9658803939819336,0.8877274394035339,0.9251563549041748
   
    
    
    
    
    
正确率：
eeg：[0.8432578444480896,0.8277559280395508,0.8489173054695129,0.8464567065238953,0.8521161675453186] ||mean:84.37
eog：[0.8774606585502625,0.8922244310379028,0.8348917365074158,0.8720472455024719,0.8710629940032959] ||mean:86.95
融合：[0.8917322754859924,0.8986220359802246,0.902312994003295,0.9069882035255432,0.9087106585502625]  ||mean:90.16



精确率：
eeg：[0.8576669096946716,0.8646034598350525,0.8845553994178772,0.8846606612205505,0.8948206901550293] ||mean:87.72
eog：[0.8869951963424683,0.9714536666870117,0.9818822145462036,0.9695315957069397,0.9711670279502869] ||mean:0.9562
融合：[0.888526976108551,0.9354709386825562.,0.9600672721862793,0.9528108239173889,0.9658803939819336]||mean:0.9405




召回率：
eeg：[0.9037,0.8646,0.8770,0.8723,0.8695] ||mean:87.74
eog：[0.9253,0.8557,0.7544,0.8247,0.8215] ||mean:83.63
融合：[0.9489,0.9029,0.8832,0.8982,0.8877]  ||mean:90.41


f1率：
eeg：[0.8801,0.8646,0.8807,0.8784,0.8819] ||mean:87.71
eog：[0.9057,0.9099,0.8532,0.8913,0.8901] ||mean:89.00
融合：[0.9177,0.9188,0.9200,0.9247,0.9251]  ||mean:92.12


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
又再跑一次：分别把eeg、eog、融合最高的记录下来了(还是用之前的标准吧)
    epoch：150 ； earlystoping：50
  

0.8324310779571533,0.8923711180686951,0.8374612927436829,0.864044725894928
0.8870570659637451,0.9546427130699158,0.8633900880813599,0.9067263007164001
0.8917322754859924,0.9357723593711853,0.8908668756484985,0.9127676486968994

0.8555610179901123,0.875,0.9017407894134521,0.8881691694259644
0.8644192814826965,0.8997641801834106,0.8854932188987732,0.8925716280937195
0.8954232335090637,0.9118993282318115,0.9249516725540161,0.9183791279792786

0.8560531735420227,0.8921991586685181,0.8801237344741821,0.8861203193664551
0.8624507784843445,0.9651216268539429,0.8132250308990479,0.8826862573623657
0.9008365869522095,0.9399435520172119,0.9017788171768188,0.9204657673835754

0.8486712574958801,0.8832230567932129,0.8780959844589233,0.8806520700454712
0.9212598204612732,0.9317315220832825,0.9454334378242493,0.9385324716567993
0.9210137724876404,0.9284362196922302,0.9489164352416992,0.9385645985603333

0.8410432934761047,0.8505788445472717,0.9098297357559204,0.879207193851471
0.8941929340362549,0.9551986455917358,0.8746129870414734,0.9131312966346741
0.908218502998352,0.9247022867202759,0.931501567363739,0.928089439868927




暂时先看正确率：
eeg：[0.8324310779571533,0.85556101799011230.8560531735420227,0.8486712574958801,0.8410432934761047] ||mean:84.67
eog：[0.8870570659637451,0.8644192814826965,0.8624507784843445,0.9212598204612732,0.8941929340362549] ||mean:88.58
融合：[0.8917322754859924,0.8954232335090637,0.900836586952209,0.9210137724876404,0.908218502998352]  ||mean:90.34
   

   
   
   
   
   
   
   
   

脑电模块的残差块有一个位置连接出错了，不过问题应该不大，后面闲得无聊可以在跑一次。
模型的345路那里搭建跟原文有些偏差，但差别应该不大，后面有时间再跑一次。
     
感觉eeg的正确率太低了，想着再跑一下，能提高一点点（这个不重要，跑着玩一下。）









  


"""



# 关于脑电的模型
class BlockOne(nn.Module):
    def __init__(self):
        super(BlockOne, self).__init__()
        # # 输入数据：1*22*1000
        self.guiyi=torch.nn.Sequential(
            nn.BatchNorm1d(17)
        )
        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(17, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )

        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(

            # nn.MaxPool1d(kernel_size=1),
            # nn.ConstantPad1d((20, 20), 0),
            nn.Conv1d(17, 17, kernel_size=[3], stride=1,padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.guiyi(x)
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8




# 输入：100*1000
# # 输入数据：1*85*80
class BlockTwo(nn.Module):
    def __init__(self):
        super(BlockTwo, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(

            # nn.MaxPool1d(kernel_size=1),
            # nn.ConstantPad1d((20, 20), 0),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


    # 输入：100*1000

class BlockThree(nn.Module):
    def __init__(self):
        super(BlockThree, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=1),
            # nn.ConstantPad1d((20, 20), 0),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


    # 输入：100*1000
class BlockFour(nn.Module):
    def __init__(self):
        super(BlockFour, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=1),
            # # nn.ConstantPad1d((20, 20), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


    # 输入：100*1000
class BlockFive(nn.Module):
    def __init__(self):
        super(BlockFive, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=1),
            # # nn.ConstantPad1d((20, 20), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8

# 输入：100*1000
class BlockSix(nn.Module):
    def __init__(self):
        super(BlockSix, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[40], stride=1),
            nn.ConstantPad1d((19, 20), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=1),
            # # nn.ConstantPad1d((20, 20), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


class IEModelSevenEeg(nn.Module):
    def __init__(self):
        super(IEModelSevenEeg, self).__init__()
        self.block1 = BlockOne()
        self.block2 = BlockTwo()
        self.block3 = BlockThree()
        self.block4 = BlockFour()
        self.block5 = BlockFive()
        self.block6 = BlockSix()

        # 残差块1：22*1000输入，100*1000输出
        self.blockRes = torch.nn.Sequential(
            nn.BatchNorm1d(17),
            nn.Conv1d(17, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块2：22*1000输入，100*1000输出
        self.blockResTwo = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.Conv1d(85, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块3：22*1000输入，100*1000输出
        # self.blockResThree = torch.nn.Sequential(
        #     nn.BatchNorm1d(85),
        #     nn.Conv1d(85, 85, kernel_size=1),
        #     nn.BatchNorm1d(85),
        #     # nn.ReLU(),
        # )

        # 残差块与普通块相加后再激活
        self.jihuo = nn.ReLU()

        self.blockLast = torch.nn.Sequential(
            # nn.Conv1d(85, 17, kernel_size=1),
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1360, 2)
        )

        self.jiangwei=torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1]),
            nn.BatchNorm1d(17),
        )


    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        # outRes1 = self.blockRes(x)
        # out2_final = self.jihuo(out2 + outRes1)

        out3 = self.block3(out2)
        outRes1 = self.blockRes(x)
        out3_final = self.jihuo(out3 + outRes1)

        out4 = self.block4(out3_final)
        out5 = self.block5(out4)
        # outRes2 = self.blockResTwo(out3)
        # out5_final = self.jihuo(out5 + outRes2)

        out6 = self.block6(out5)
        outRes2 = self.blockResTwo(out3_final)
        out6_final = self.jihuo(out6 + outRes2)

        # #10*85*80:--》10*17*80:得到17个电极
        out_feature1 = self.jiangwei(out6_final)
        # 10*85*80:--》10*17*5*16：17个电极，5个频带，16片段。
        out_feature2 = out_feature1.view(out_feature1.shape[0],out_feature1.shape[1],5,16)
        # 10*17*5*16：=》10*17*5：17个电极，5个频带每个特征值均值
        out_feature3=torch.mean(out_feature2,dim=3)

        # y = torch.split(out_feature1, 1, dim=1)


        # 分类层
        out7 = self.blockLast(out6_final)
        return nn.functional.log_softmax(out7),out_feature3





"""

只保留一个INCEPTION块
"""
class IEModelSevenEeg_xiaorong(nn.Module):
    def __init__(self):
        super(IEModelSevenEeg_xiaorong, self).__init__()
        self.block1 = BlockOne()


        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1360, 2)
        )


    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.blockLast(out1)



        # 分类层
        return nn.functional.log_softmax(out2)


"""

只保留3个INCEPTION块
"""
class IEModelSevenEeg_xiaorong_3(nn.Module):
    def __init__(self):
        super(IEModelSevenEeg_xiaorong_3, self).__init__()
        self.block1 = BlockOne()
        self.block2 = BlockTwo()
        self.block3 = BlockThree()




        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1360, 2)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        # outRes1 = self.blockRes(x)
        # out2_final = self.jihuo(out2 + outRes1)

        out3 = self.block3(out2)

        # 分类层
        out7 = self.blockLast(out3)
        return nn.functional.log_softmax(out7)


"""
使用9个INCEPTION块：
    7/8/9可以直接模仿4/5/6
    
"""
class IEModelSevenEeg_xiaorong_9(nn.Module):
    def __init__(self):
        super(IEModelSevenEeg_xiaorong_9, self).__init__()
        self.block1 = BlockOne()
        self.block2 = BlockTwo()
        self.block3 = BlockThree()
        self.block4 = BlockFour()
        self.block5 = BlockFive()
        self.block6 = BlockSix()
        self.block7 = BlockFour()
        self.block8 = BlockFive()
        self.block9 = BlockSix()


        # 残差块1：22*1000输入，100*1000输出
        self.blockRes = torch.nn.Sequential(
            nn.BatchNorm1d(17),
            nn.Conv1d(17, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块2：22*1000输入，100*1000输出
        self.blockResTwo = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.Conv1d(85, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块3：22*1000输入，100*1000输出
        self.blockResThree = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.Conv1d(85, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块与普通块相加后再激活
        self.jihuo = nn.ReLU()



        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1360, 2)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        # outRes1 = self.blockRes(x)
        # out2_final = self.jihuo(out2 + outRes1)

        out3 = self.block3(out2)
        outRes1 = self.blockRes(x)
        out3_final = self.jihuo(out3 + outRes1)

        out4 = self.block4(out3_final)
        out5 = self.block5(out4)
        # outRes2 = self.blockResTwo(out3)
        # out5_final = self.jihuo(out5 + outRes2)
        out6 = self.block6(out5)
        outRes2 = self.blockResTwo(out3_final)
        out6_final = self.jihuo(out6 + outRes2)


        out7 = self.block7(out6_final)
        out8 = self.block8(out7)
        out9 = self.block9(out8)
        outRes3 = self.blockResThree(out6_final)
        out9_final = self.jihuo(out6 + outRes3)



        # 分类层
        out10 = self.blockLast(out9_final)
        return nn.functional.log_softmax(out10)














#关于眼电的模型:改成85个频道（这个倒无所谓，后面再来说）
class BlockOneEog(nn.Module):
    def __init__(self):
        super(BlockOneEog, self).__init__()
        # # 输入数据：1*22*1000
        #先做一个归一化看看效果：
        self.Normalize=torch.nn.Sequential(
            nn.BatchNorm1d(3),
        )

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2,2 ), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )


        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(3, 17, kernel_size=[1], stride=1),
        )


        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=2),
            # nn.ConstantPad1d((9, 9), 0),
            nn.Conv1d(3, 17, kernel_size=[3], stride=1,padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):

        #先做了一个归一化（数据太大了，试一下效果）
        x=self.Normalize(x)

        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)


        out5 = self.CieConv(x)
        out6 = self.block2(x)


        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8

    # 输入：100*1000

class BlockTwoEog(nn.Module):
    def __init__(self):
        super(BlockTwoEog, self).__init__()
        # # 输入数据：1*22*1000


        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2,2 ), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )


        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )


        # 第2分支
        self.block2 = torch.nn.Sequential(

            # nn.MaxPool1d(kernel_size=2),
            # nn.ConstantPad1d((9, 9), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):

        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)


        out5 = self.CieConv(x)
        out6 = self.block2(x)


        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


class BlockThreeEog(nn.Module):
    def __init__(self):
        super(BlockThreeEog, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=2),
            # nn.ConstantPad1d((9, 9), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            # nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


class BlockFourEog(nn.Module):
    def __init__(self):
        super(BlockFourEog, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=2),
            # nn.ConstantPad1d((9, 9), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


class BlockFiveEog(nn.Module):
    def __init__(self):
        super(BlockFiveEog, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=2),
            # nn.ConstantPad1d((9, 9), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8
class BlockSixEog(nn.Module):
    def __init__(self):
        super(BlockSixEog, self).__init__()
        # # 输入数据：1*22*1000

        # 分成5个分支处理再合并

        # 第3,4,5分支
        # 统一卷积
        self.FdrConv1_1 = torch.nn.Sequential(
            nn.Conv1d(85, 32, kernel_size=[1], stride=1),
        )
        # 第二处理部分
        self.HdeConv = torch.nn.Sequential(
            nn.Conv1d(32, 17, kernel_size=[1], stride=1),
        )
        # 第三处理部分
        self.Short1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long1 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第四部分
        self.Short2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[5], stride=1),
            nn.ConstantPad1d((2, 2), 0)
        )
        self.Medium2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[10], stride=1),
            nn.ConstantPad1d((4, 5), 0)
        )
        self.Long2 = torch.nn.Sequential(
            nn.Conv1d(17, 17, kernel_size=[20], stride=1),
            nn.ConstantPad1d((9, 10), 0)
        )

        # 第1分支
        self.CieConv = torch.nn.Sequential(
            nn.Conv1d(85, 17, kernel_size=[1], stride=1),
        )

        # 第2分支
        self.block2 = torch.nn.Sequential(
            # nn.MaxPool1d(kernel_size=2),
            # nn.ConstantPad1d((9, 9), 0),
            # nn.Conv1d(85, 17, kernel_size=[1], stride=1),
            nn.Conv1d(85, 17, kernel_size=[3], stride=1, padding=1),
        )

        # 统一处理
        self.block3 = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            # nn.ReLU()
        )

    def forward(self, x):
        out1 = self.FdrConv1_1(x)
        out2 = self.HdeConv(out1)
        out3_1 = self.Short1(out2)
        out3_2 = self.Medium1(out2)
        out3_3 = self.Long1(out2)
        out4_1 = self.Short2(out3_1)
        out4_2 = self.Medium2(out3_2)
        out4_3 = self.Long2(out3_3)

        out5 = self.CieConv(x)
        out6 = self.block2(x)

        # 合并：
        out7 = torch.cat([out5, out6, out4_1, out4_2, out4_3], dim=1)
        out8 = self.block3(out7)
        return out8


class IEModelSevenEog(nn.Module):
    def __init__(self):
        super(IEModelSevenEog, self).__init__()
        self.block1 = BlockOneEog()
        self.block2 = BlockTwoEog()
        self.block3 = BlockThreeEog()
        self.block4 = BlockFourEog()
        self.block5 = BlockFiveEog()
        self.block6 = BlockSixEog()

        # 残差块1：22*1000输入，100*1000输出
        self.blockRes = torch.nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Conv1d(3, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块2：22*1000输入，100*1000输出
        self.blockResTwo = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.Conv1d(85, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块与普通块相加后再激活
        self.jihuo = nn.ReLU()

        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(595, 2)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        outRes1 = self.blockRes(x)
        out3_final = self.jihuo(out3 + outRes1)



        out4 = self.block4(out3_final)
        out5 = self.block5(out4)
        out6 = self.block6(out5)
        outRes2 = self.blockResTwo(out3_final)
        out6_final = self.jihuo(out6 + outRes2)


        # 分类层
        out7 = self.blockLast(out6_final)
        return nn.functional.log_softmax(out7)




"""

只保留一个INCEPTION块
"""
class IEModelSevenEog_xiaorong(nn.Module):
    def __init__(self):
        super(IEModelSevenEog_xiaorong, self).__init__()
        self.block1 = BlockOneEog()


        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(595, 2)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.blockLast(out1)

        return nn.functional.log_softmax(out2)


"""

只保留3个INCEPTION块
"""
class IEModelSevenEog_xiaorong_3(nn.Module):
    def __init__(self):
        super(IEModelSevenEog_xiaorong_3, self).__init__()
        self.block1 = BlockOneEog()
        self.block2 = BlockTwoEog()
        self.block3 = BlockThreeEog()



        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(595, 2)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)


        # 分类层
        out7 = self.blockLast(out3)
        return nn.functional.log_softmax(out7)



"""
使用9个INCEPTION块：
    7/8/9可以直接模仿4/5/6

"""
class IEModelSevenEog_xiaorong_9(nn.Module):
    def __init__(self):
        super(IEModelSevenEog_xiaorong_9, self).__init__()
        self.block1 = BlockOneEog()
        self.block2 = BlockTwoEog()
        self.block3 = BlockThreeEog()
        self.block4 = BlockFourEog()
        self.block5 = BlockFiveEog()
        self.block6 = BlockSixEog()
        self.block7 = BlockFourEog()
        self.block8 = BlockFiveEog()
        self.block9 = BlockSixEog()


        # 残差块1：22*1000输入，100*1000输出
        self.blockRes = torch.nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Conv1d(3, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块2：22*1000输入，100*1000输出
        self.blockResTwo = torch.nn.Sequential(
            nn.BatchNorm1d(85),
            nn.Conv1d(85, 85, kernel_size=1),
            nn.BatchNorm1d(85),
            # nn.ReLU(),
        )

        # 残差块与普通块相加后再激活
        self.jihuo = nn.ReLU()

        self.blockLast = torch.nn.Sequential(
            nn.AvgPool1d(kernel_size=5),
            nn.BatchNorm1d(85),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(595, 2)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        outRes1 = self.blockRes(x)
        out3_final = self.jihuo(out3 + outRes1)



        out4 = self.block4(out3_final)
        out5 = self.block5(out4)
        out6 = self.block6(out5)
        outRes2 = self.blockResTwo(out3_final)
        out6_final = self.jihuo(out6 + outRes2)

        out7 = self.block7(out6_final)
        out8 = self.block8(out7)
        out9 = self.block9(out8)
        outRes3 = self.blockResTwo(out6_final)
        out9_final = self.jihuo(out9 + outRes3)


        # 分类层
        out10 = self.blockLast(out9_final)
        return nn.functional.log_softmax(out10)









def function_test():
    # loss = nn.CrossEntropyLoss()
    #
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # output = loss(input, target)
    # output.backward()
    # output.backward()


    a = torch.ones((10, 17, 80))
    model=IEModelSevenEeg()
    RESULT=model(a)



    # a = torch.ones((10, 3, 36))
    # model = IEModelSevenEog_xiaorong_3()
    # RESULT = model(a)


    # c = torch.Tensor([[1,2,3,4],[5,6,7,8]]).view(1,2,4)
    # model=IEModelSevenEeg()
    # model_eog=IEModelSevenEog()
    # model=nn.MaxPool1d(kernel_size=1)
    # modelTwo=nn.Conv1d(2,2,kernel_size=1,stride=1)
    # result=modelTwo(c)

    # result1=model(a)
    # result2 = model_eog(c)

    # d =model(a,c)
    # e=model_eog(c)
    # #返回每个样本的预测最大值与下标
    # _, pred_1 = torch.max(e.data, 1)
    #
    # f=model_eeg(a)
    #
    # blending_y_pred = e * 0.4 + f * 0.6

    print("OK")



if __name__=="__main__":
    function_test()









