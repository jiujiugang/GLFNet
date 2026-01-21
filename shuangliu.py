from os import path
import os
import numpy as np
import cv2
import time
import pandas as pd

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch
from model_start_4_new import FusionModel
import numpy as np
from facenet_pytorch import MTCNN
# Some of the codes are adapted from STSNet
import warnings
warnings.filterwarnings("ignore")
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):  #这段代码定义了一个函数 confusionMatrix，用于计算混淆矩阵的相关指标，包括 F1 分数和平均召回率
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):#定义一个函数 recognition_evaluation，它接受三个参数：final_gt（真实的情感标签），final_pred（模型预测的情感标签），以及一个可选参数 show
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}#情感标签映射到数字编码
    # Display recognition result
    f1_list = []
    ar_list = []#初始化了两个空列表，用于存储每个情感类别的F1分数和识别率。
    try:
        for emotion, emotion_index in label_dict.items():#遍历 label_dict 字典的键值对，emotion 表示情感标签的字符串形式，emotion_index 表示对应的数值形式。
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]#使用列表推导式生成两个新列表 gt_recog 和 pred_recog
        #对于 final_gt 和 final_pred 中的每个元素，如果它等于当前的 emotion_index，则在新列表中标记为 1（表示该样本属于当前情感类别），否则标记为 0。
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)#将计算得到的F1分数和识别率添加到相应的列表中。
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR#计算所有情感类别的平均F1分数和平均识别率。
    except:
        return '', ''#如果发生异常，返回两个空字符串。

def whole_face_block_coordinates():  # 根据图像中的人脸特征提取出人脸块的中心坐标。
        df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
        m, n = df.shape
        base_data_src = './datasets/combined_datasets_whole'
        total_emotion = 0
        image_size_u_v = 28
        # get the block center coordinates
        face_block_coordinates = {}

        # for i in range(0, m):
        for i in range(0, m):  # 一个for循环，它将从0开始迭代到m-1
            image_name = str(df['sub'][i]) + '_' + str(
                df['filename_o'][i]) + ' .png'  # 这行代码拼接字符串以生成一个图像名称它从数据框的'sub'和'filename_o'列中取出相应行的值，并加上'.png'后缀。
            # print(image_name)
            img_path_apex = base_data_src + '/' + df['imagename'][
                i]  # 这行代码构建了一个图像文件的完整路径。它将基础数据源路径base_data_src与数据框中'imagename'列的值拼接起来
            train_face_image_apex = cv2.imread(img_path_apex)  # (444, 533, 3)
            face_apex = cv2.resize(train_face_image_apex, (28, 28),
                                   interpolation=cv2.INTER_AREA)  # 这行代码将读取的图像大小调整为28x28像素，使用INTER_AREA插值方法。
            # get face and bounding box
            mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
            # 初始化一个MTCNN对象，用于人脸检测和关键点定位。这里设置了几个参数，包括人脸检测框的边界大小（margin）、输入图像的大小（image_size_u_v）、是否只选择最大的人脸（select_largest）、是否进行后处理（post_process）以及使用的设备（device，这里是CUDA设备0，即第一个GPU）。
            batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
            # 使用MTCNN对象检测调整大小后的图像中的人脸和关键点。返回的batch_boxes是包含人脸边界框的数组，batch_landmarks是包含人脸关键点坐标的数组
            # print(img_path_apex,batch_landmarks)
            # if not detecting face
            if batch_landmarks is None:
                # print( df['imagename'][i])
                batch_landmarks = np.array([[[9.528073, 11.062551]
                                                , [21.396168, 10.919773]
                                                , [15.380184, 17.380562]
                                                , [10.255435, 22.121233]
                                                , [20.583706, 22.25584]]])  # 如果MTCNN没有检测到关键点，这行代码会手动设置一组默认的关键点坐标。
                # print(img_path_apex)
            row_n, col_n = np.shape(batch_landmarks[0])  # 获取关键点数组的行数和列数。
            # print(batch_landmarks[0])
            for i in range(0, row_n):
                for j in range(0, col_n):  # 嵌套循环，用于遍历每个关键点的坐标。
                    if batch_landmarks[0][i][j] < 7:
                        batch_landmarks[0][i][j] = 7
                    if batch_landmarks[0][i][j] > 21:
                        batch_landmarks[0][i][j] = 21  # 这两行代码检查每个关键点的坐标是否小于7或大于21，如果是，则进行坐标调整。
            batch_landmarks = batch_landmarks.astype(int)  # 将batch_landmarks数组中的浮点数坐标转换为整数
            # print(batch_landmarks[0])
            # get the block center coordinates
            face_block_coordinates[image_name] = batch_landmarks[
                0]  # 将检测到的关键点坐标（batch_landmarks[0]）存储在一个名为face_block_coordinates的字典中
        # print(len(face_block_coordinates))
        return face_block_coordinates
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()#获取了人脸块的坐标信息，并将其存储在变量中
    # print(len(face_block_coordinates_dict))
    # Get train dataset
    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'#定义了包含整个面部光流图像的文件夹路径。
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)#使用 os.listdir 函数列出 whole_optical_flow_path 路径下的所有文件名，并存储在 whole_optical_flow_imgs 列表中
    four_parts_optical_flow_imgs = {}#初始化一个空字典 four_parts_optical_flow_imgs，用于存储裁剪后的局部光流图像
    # print(whole_optical_flow_imgs[0]) #spNO.189_f_150.png
    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)#使用 OpenCV 的 imread 函数读取当前光流图像，并存储在 flow_image 变量中。
        four_part_coordinates = face_block_coordinates_dict[n_img]#从 face_block_coordinates_dict 字典中获取当前图像 n_img 的面部关键点坐标。
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                 four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
        # print(np.shape(l_eye))这些行使用面部关键点坐标从 flow_image 中裁剪出左眼、左唇、鼻子、右眼和右唇的局部光流图像，并将它们添加到 four_parts_optical_flow_imgs[n_img] 列表中
    # print((four_parts_optical_flow_imgs['spNO.189_f_150.png'][0]))->(14,14,3)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs
def all_optical_flow_block():
    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'  # 定义包含整个面部光流图像的文件夹路径。
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)  # 列出路径下的所有文件名
    global_optical_flow_dict = {}  # 初始化一个空字典，用于存储全局光流图像

    for n_img in whole_optical_flow_imgs:
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)  # 读取当前光流图像
        global_optical_flow_dict[n_img] = flow_image  # 将图像添加到字典中


    return global_optical_flow_dict  # 返回字典



def main(config):
    learning_rate = 0.00005
    batch_size =64
    epochs =60
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss()
    if config.train:
        if not path.exists('ourmodel_threedatasets_weight22'):
            os.mkdir('ourmodel_threedatasets_weight22')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()

    main_path = './datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    global_optical_flow = all_optical_flow_block()
    print("global_optical_flow keys:", global_optical_flow.keys())
    print("subName:", subName)
    print("global_optical_flow keys:", global_optical_flow.keys())

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []
        global_train = []
        global_test = []

        # Get train dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))
                if n_img not in global_optical_flow:
                    print(f"Error: {n_img} not found in global_optical_flow")
                global_train.append(global_optical_flow[n_img])
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_train.append(lr_eye_lips)

        # Get test dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)
            for n_img in img:
                y_test.append(int(n_expression))
                if n_img not in global_optical_flow:
                    print(f"Error: {n_img} not found in global_optical_flow")
                global_test.append(global_optical_flow[n_img])
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)

        weight_path = 'ourmodel_threedatasets_weight22' + '/' + n_subName + '.pth'

        # 将模型初始化并传输到设备
        model = FusionModel(num_classes=3)
        model = model.to(device)

        if config.train:
            print('train')
        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 将数据转换为Tensor并创建DataLoader
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        four_parts_train = torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)
        global_train = torch.Tensor(np.array(global_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(global_train, four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,drop_last=False )

        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        global_test = torch.Tensor(np.array(global_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(global_test, four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        # 记录最佳结果
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if config.train:
                model.train()#初始化
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:#从 train_dl (训练数据加载器) 中获取一个批次的训练数据。
                    optimizer.zero_grad()
                    global_x = batch[0].to(device)
                    parts_x = batch[1].to(device)
                    y = batch[2].to(device)

                    #print("global_x size:", global_x.size())
                    #print("parts_x size:", parts_x.size())

                    yhat = model(global_x, parts_x)#通过模型计算预测值 yhat。
                    loss = loss_fn(yhat, y)#计算预测值 yhat 和真实标签 y 之间的损失 loss
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * global_x.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += global_x.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)#累积当前批次的损失和正确预测的数量。

            model.eval()#将模型设为评估模式,这会禁用dropout和batch normalization等。
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                global_x = batch[0].to(device)
                parts_x = batch[1].to(device)
                y = batch[2].to(device)
                yhat = model(global_x, parts_x)
                loss = loss_fn(yhat, y)
                val_loss += loss.data.item() * global_x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)

            temp_best_each_subject_pred = []#初始化临时变量 temp_best_each_subject_pred 用于存储当前批次的最佳预测结果。
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred#更新最佳准确率
                if config.train:
                    torch.save(model.state_dict(), weight_path)#保存模型

        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y.tolist()
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True)
    config = parser.parse_args()
    main(config)
