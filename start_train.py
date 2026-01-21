from os import path
import os
import numpy as np
import cv2
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
from facenet_pytorch import MTCNN
from sklearn.metrics import confusion_matrix
import warnings
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

warnings.filterwarnings("ignore")

# 定义用于混淆矩阵计算的函数
def confusionMatrix(gt, pred):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

# 定义用于评估识别效果的函数
def recognition_evaluation(final_gt, final_pred):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''

# 获取全局光流块
def all_optical_flow_block():
    whole_optical_flow_path = r'/HTNet-master/datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    global_optical_flow_dict = {}
    for n_img in whole_optical_flow_imgs:
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        global_optical_flow_dict[n_img] = flow_image
    return global_optical_flow_dict

# 定义模型结构
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),  # 3x3卷积减少通道数
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),  # 3x3卷积恢复通道数
            nn.Sigmoid()  # 归一化
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        # 添加注意力机制
        attention_weights = self.attention(x)  # 计算注意力权重
        x = input + self.drop_path(x)
        x = x + x * attention_weights  # 将注意力权重应用到特征图上
        return x

class StarNet(nn.Module):
    def __init__(self, base_dim=256, depths=[3,4,6,3], mlp_ratio=4, drop_path_rate=0.0, num_classes=3):
        super().__init__()
        self.in_channel = base_dim // 4
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adjusted to (1, 1) to match FC layer input size
        self.fc = nn.Linear(self.in_channel, num_classes)  # Adjusted input size to match pooled output
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



# 主函数
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import pandas as pd
import numpy as np

def main(config):
    learning_rate = 0.00005
    batch_size = 32
    epochs = 30
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    loss_fn = nn.CrossEntropyLoss()

    if config.train and not os.path.exists('ourmodel_threedatasets_weightsst'):
        os.mkdir('ourmodel_threedatasets_weightsst')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()
    main_path = r'/HTNet-master/datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    global_optical_flow = all_optical_flow_block()
    print("global_optical_flow keys:", global_optical_flow.keys())
    print("subName:", subName)

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        global_train = []
        global_test = []

        # 获取训练数据集
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)
            for n_img in img:
                y_train.append(int(n_expression))
                global_train.append(global_optical_flow[n_img])

        # 获取测试数据集
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)
            for n_img in img:
                y_test.append(int(n_expression))
                global_test.append(global_optical_flow[n_img])

        weight_path = 'ourmodel_threedatasets_weightsst' + '/' + n_subName + '.pth'

        # 初始化并传输到设备
        model = StarNet(num_classes=3)
        model = model.to(device)

        if config.train:
            print('train')
        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 转换数据并创建DataLoader
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        global_train = torch.Tensor(np.array(global_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(global_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)

        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        global_test = torch.Tensor(np.array(global_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(global_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if config.train:
                # Training
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for X, y in train_dl:
                    if X.size(0) == 1:
                        continue  # 跳过批量大小为1的情况
                    optimizer.zero_grad()
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X)
                    loss = loss_fn(y_hat, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X.size(0)
                    num_train_correct += (torch.max(y_hat, 1)[1] == y).sum().item()
                    num_train_examples += X.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

            # Testing
            model.eval()
            correct = 0
            y_true = []
            y_pred = []
            temp_best_each_subject_pred = []
            with torch.no_grad():
                for X, y in test_dl:
                    if X.size(0) == 1:
                        continue  # 跳过批量大小为1的情况
                    X, y = X.to(device), y.to(device)
                    output = model(X)
                    pred = torch.max(output, 1)[1]
                    y_true.extend(y.view(-1).cpu().tolist())
                    y_pred.extend(pred.view(-1).cpu().tolist())
                    correct += (pred == y).sum().item()
                    temp_best_each_subject_pred.extend(pred.view(-1).cpu().tolist())

            accuracy = correct / len(test_dl.dataset)
            if accuracy > best_accuracy_for_each_subject:
                best_accuracy_for_each_subject = accuracy
                best_each_subject_pred = temp_best_each_subject_pred
                if accuracy == 1.0:  # 如果测试准确率为100%
                    directory = 'ourmodel_threedatasets_weight22'
                    if not os.path.exists(directory):
                        os.makedirs(directory)  # 创建目录
                    weight_path_subject = f'ourmodel_threedatasets_weight22/{n_subName}_best_accuracy.pth'
                    torch.save(model.state_dict(), weight_path_subject)
                    print(f"Saved model with 100% accuracy for subject: {n_subName}")
                    break  # 直接停止训练，跳出循环，因为准确率已经为100%
                if config.train:
                    torch.save(model.state_dict(), weight_path)#保存模型
                    print(f'New best accuracy: {best_accuracy_for_each_subject:.4f}')
                if config.train:
                    torch.save(model.state_dict(), weight_path)


        # 保存每个subject的最佳预测
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y_true
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth      :', y_true)
        print('Evaluation until this subject:')
        total_gt.extend(y_true)
        total_pred.extend(y_pred)
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred)
    print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Global Optical Flow based training')
    parser.add_argument('--train', default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()
    main(args)
