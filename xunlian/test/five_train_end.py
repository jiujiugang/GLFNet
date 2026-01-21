import os
import numpy as np
import cv2
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
from Modelor import FusionModel
from facenet_pytorch import MTCNN
import warnings

warnings.filterwarnings("ignore")

# 重置模型权重
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# 计算混淆矩阵并计算F1分数和平均召回率
def confusionMatrix(gt, pred, show=False):
    try:
        TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
        f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0.0
        num_samples = len([x for x in gt if x == 1])
        average_recall = TP / num_samples if num_samples != 0 else 0.0
        return f1_score, average_recall
    except ValueError:  # 处理混淆矩阵计算失败的情况
        return 0.0, 0.0

# 评估识别结果
def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'Anger': 0, 'Happiness': 1, 'Contempt': 2, 'Other': 3, 'Surprise': 4}
    f1_list = []
    ar_list = []
    for emotion, emotion_index in label_dict.items():
        gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
        pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
        f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
        f1_list.append(f1_recog)
        ar_list.append(ar_recog)
    UF1 = np.mean(f1_list)
    UAR = np.mean(ar_list)
    return UF1, UAR

# 获取面部块坐标
def whole_face_block_coordinates():
    df = pd.read_csv('../five_samm.csv')
    m, n = df.shape
    base_data_src = r'D:\HTNet-master\HTNet-master\datasets\samm_apex'  # 读取图片顶点帧
    image_size_u_v = 28
    face_block_coordinates = {}

    for i in range(m):
        image_name = str(df['imagename'][i])
        image_name = image_name.replace('.jpg', '.png')
        img_path_apex = os.path.join(base_data_src, df['imagename'][i])
        train_face_image_apex = cv2.imread(img_path_apex)

        # 检查图像是否加载成功
        if train_face_image_apex is None:
            print(f"Warning: Image {img_path_apex} not loaded successfully.")
            continue

        try:
            face_apex = cv2.resize(train_face_image_apex, (image_size_u_v, image_size_u_v),
                                   interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            print(f"Error resizing image {img_path_apex}: {e}")
            continue

        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)

        if batch_landmarks is None:
            print(f"No landmarks detected for image {img_path_apex}.")
            batch_landmarks = np.array([[[9.528073, 11.062551], [21.396168, 10.919773], [15.380184, 17.380562],
                                         [10.255435, 22.121233], [20.583706, 22.25584]]])

        # 保证坐标值在合理范围内
        row_n, col_n = np.shape(batch_landmarks[0])
        for i in range(row_n):
            for j in range(col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        face_block_coordinates[image_name] = batch_landmarks[0]

    return face_block_coordinates

# 裁剪光流块
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()
    whole_optical_flow_path = r'D:\HTNet-master\HTNet-master\datasets\SAMM_five_class_optical'  # 定义包含整个面部光流图像的文件夹路径。
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}

    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img] = []
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        # 遍历图片列表
        l_eye = flow_image[four_part_coordinates[0][0] - 7:four_part_coordinates[0][0] + 7,
                 four_part_coordinates[0][1] - 7: four_part_coordinates[0][1] + 7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                 four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].extend([l_eye, l_lips, nose, r_eye, r_lips])

    return four_parts_optical_flow_imgs

# 获取所有光流块
def all_optical_flow_block():
    whole_optical_flow_path = r'D:\HTNet-master\HTNet-master\datasets\SAMM_five_class_optical'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    global_optical_flow_dict = {}

    for n_img in whole_optical_flow_imgs:
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        global_optical_flow_dict[n_img] = flow_image

    return global_optical_flow_dict

# 主函数
def main(config):
    # 模型和训练设置
    learning_rate = 0.00005
    batch_size = 32
    epochs = 60
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    loss_fn = nn.CrossEntropyLoss()

    if config.train and not os.path.exists('../ourmodel_threedatasets_weightfive'):
        os.mkdir('../ourmodel_threedatasets_weightfive')

    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Device: {device}\n')

    # 初始化性能记录变量
    total_gt = []
    total_pred = []
    all_accuracy_dict = {}

    start_time = time.time()

    main_path = r'D:\HTNet-master\HTNet-master\datasets\SAMM_five'
    subNames = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    global_optical_flow = all_optical_flow_block()

    for n_subName in subNames:
        print(f'Subject: {n_subName}')
        y_train, y_test = [], []
        four_parts_train, four_parts_test = [], []
        global_train, global_test = [], []

        # 加载训练数据
        for n_expression in os.listdir(main_path + '/' + n_subName + '/u_train'):
            for n_img in os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression):
                y_train.append(int(n_expression))
                if n_img in global_optical_flow:
                    global_train.append(global_optical_flow[n_img])
                    l_eye_lips = cv2.hconcat(
                        [all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                    r_eye_lips = cv2.hconcat(
                        [all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                    four_parts_train.append(cv2.vconcat([l_eye_lips, r_eye_lips]))

        # 加载测试数据
        for n_expression in os.listdir(main_path + '/' + n_subName + '/u_test'):
            for n_img in os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression):
                y_test.append(int(n_expression))
                if n_img in global_optical_flow:
                    global_test.append(global_optical_flow[n_img])
                    l_eye_lips = cv2.hconcat(
                        [all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                    r_eye_lips = cv2.hconcat(
                        [all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                    four_parts_test.append(cv2.vconcat([l_eye_lips, r_eye_lips]))

        x_global_train = torch.tensor(np.transpose(global_train, (0, 3, 1, 2)), dtype=torch.float32)
        x_global_test = torch.tensor(np.transpose(global_test, (0, 3, 1, 2)), dtype=torch.float32)
        x_four_parts_train = torch.tensor(np.transpose(four_parts_train, (0, 3, 1, 2)), dtype=torch.float32)
        x_four_parts_test = torch.tensor(np.transpose(four_parts_test, (0, 3, 1, 2)), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(x_global_train, x_four_parts_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(x_global_test, x_four_parts_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = FusionModel(num_classes=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 重置模型权重
        reset_weights(model)

        if config.train:
            print(f"Training subject: {n_subName}")
            for epoch in range(epochs):
                model.train()
                total_loss, correct, total = 0, 0, 0

                for global_img, four_parts_img, labels in train_loader:
                    global_img, four_parts_img, labels = global_img.to(device), four_parts_img.to(device), labels.to(
                        device)
                    optimizer.zero_grad()
                    outputs = model(global_img, four_parts_img)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                accuracy = 100. * correct / total
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

            torch.save(model.state_dict(), f'ourmodel_threedatasets_weightfive/model_{n_subName}.pth')

        if config.test:
            print(f"Testing subject: {n_subName}")
            model.load_state_dict(torch.load(f'ourmodel_threedatasets_weightfive/model_{n_subName}.pth'))
            model.eval()
            correct, total = 0, 0
            all_labels, all_predicted = [], []

            with torch.no_grad():
                for global_img, four_parts_img, labels in test_loader:
                    global_img, four_parts_img, labels = global_img.to(device), four_parts_img.to(device), labels.to(
                        device)
                    outputs = model(global_img, four_parts_img)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())

            accuracy = 100. * correct / total
            UF1, UAR = recognition_evaluation(all_labels, all_predicted)
            print(f'Test Accuracy of {n_subName}: {accuracy:.2f}%')
            print(f'Test UF1: {UF1:.4f}, Test UAR: {UAR:.4f}')
            all_accuracy_dict[n_subName] = (accuracy, UF1, UAR)

            total_gt.extend(all_labels)
            total_pred.extend(all_predicted)

    total_accuracy = 100. * sum([x[0] for x in all_accuracy_dict.values()]) / len(all_accuracy_dict)
    total_UF1, total_UAR = recognition_evaluation(total_gt, total_pred)
    print(f'Total Accuracy: {total_accuracy:.2f}%')
    print(f'Total UF1: {total_UF1:.4f}, Total UAR: {total_UAR:.4f}')
    print(f'Total Time: {time.time() - start_time:.2f}s')

    return total_accuracy, total_UF1, total_UAR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True)
    parser.add_argument('--test', type=strtobool, default=True)
    config = parser.parse_args()
    main(config)

