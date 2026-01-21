import os
import sys
import cv2
import numpy as np
import pandas as pd
import face_recognition.api as face_recognition
import matplotlib.pyplot as plt
from tqdm import tqdm

data_root = 'E:\\paper\\medaima\\data\\original_image\\SMIC'                                      # 数据集所在的根目录 为SMIC数据集裁剪后的图片的顶点帧检测
smic_annotation_file = 'E:\\paper\\medaima\\data\\original_image\\biao\\SMIC-HS-E_annotation_orig.xlsx'                     # 数据集的注释文件名称
label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}                  # 定义类别与数字编码之间的对应关系,用一个字典表示,关键字是类别名称,值是对应的数字编码。


def get_clip_frame_paths(subject, filename, on_frame_idx, off_frame_idx):   # 获取视频中指定帧图片路径的函数
    frame_paths = []                                                        # 用于存储视频帧图片的路径
    subject = 's{}'.format(str(subject).zfill(2))                           # 将subject变量转换为's01','s02'等格式的字符串

    dir_path = os.path.join(data_root, subject, filename)
#                          SMIC_croped_Apexspotting/s01/s1_ne_01
#   构造从on_frame_idx到off_frame_idx范围内的所有帧图片的完整路径到frame_paths列表中  将图片帧的路径存储到frame_paths列表里
    for idx in range(on_frame_idx, off_frame_idx + 1):
        idx = str(idx).zfill(6)
        frame_path = os.path.join(dir_path, 'image{}.bmp'.format(idx))
        if not os.path.exists(frame_path):
            print('Fail to locate file', frame_path)
            raise Exception('The value of path was: {}'.format(frame_path))
        frame_paths.append(frame_path)
    return frame_paths

# 用于检测图片中的唯一一张人脸关键点
def detect_lmks(frame):                                # 函数接收一个frame参数,表示要处理的图片帧
    lmks = face_recognition.face_landmarks(frame)      # 使用face_recognition模块的face_landmarks函数来检测图片中的面部关键点
    return lmks[0]                                     # 这里我们假设只有一个人脸,因此返回结果lmks[0],取第一个人脸的关键点。将检测到的关键点结果作为字典返回。

# 从图像中裁剪出特定区域的cell
def get_cell(img, cell_location):
# img:原始图像
# cell_location：细胞区域的左上和右下角坐标点组成的元组
    point1, point2 = cell_location
    cell = img[point1[1]:point2[1], point1[0]:point2[0]]
    return cell

# 根据给定的人脸关键点坐标landmarks,计算出各个细胞区域的坐标框
def get_cell_locations(lmks):
    # 定义get_rect函数,可以传入中心点和宽度计算坐标框 输出为中心点坐标减去一半宽度的左坐标 和  中心点坐标加上一半宽度的右坐标
    def get_rect(center, width):
        point1 = np.array(center) - int(width / 2)
        point2 = np.array(center) + int(width / 2)
        return tuple(point1), tuple(point2)
    # 创建一个空的字典cells来存储提取的细胞区域坐标。
    # 计算上嘴唇区域的宽度作为细胞的默认宽度cell_width。
    cells = {}
    cell_width = int((lmks['top_lip'][6][0] - lmks['top_lip'][0][0]) / 2)

    key = 'top_lip'# 上嘴唇
    points = np.array(lmks[key])
    left_lip_rect = get_rect(points[0], cell_width)
    right_lip_rect = get_rect(points[6], cell_width)
    cells['left_lip'] = left_lip_rect
    cells['right_lip'] = right_lip_rect
    # 将得到的左右细胞坐标框存入cells字典中。

    key = 'chin' # 下巴
    point = lmks[key][int(len(lmks[key]) / 2)]
    rect_point1 = (point[0] - int(cell_width / 2), point[1] - cell_width)
    rect_point2 = (point[0] + int(cell_width / 2), point[1])
    chin_rect = (rect_point1, rect_point2)
    # 将提取的下巴细胞坐标框存入cells字典
    cells['chin_rect'] = chin_rect

    key = 'nose_tip'
    point = lmks[key][0]
    left_nose_rect_point1 = (point[0] - cell_width, left_lip_rect[0][1] - cell_width)
    left_nose_rect_point2 = (point[0], left_lip_rect[0][1])
    left_nose_rect = (left_nose_rect_point1, left_nose_rect_point2)
    cells['left_nose'] = left_nose_rect

    point = lmks[key][4]
    right_nose_rect_point1 = (point[0], right_lip_rect[0][1] - cell_width)
    right_nose_rect_point2 = (point[0] + cell_width, right_lip_rect[0][1])
    right_nose_rect = (right_nose_rect_point1, right_nose_rect_point2)
    cells['right_nose'] = right_nose_rect

    key = 'left_eye'
    point = lmks[key][0]
    left_eye_rect_point1 = (point[0] - cell_width, int(point[1] - cell_width / 2))
    left_eye_rect_point2 = (point[0], int(point[1] + cell_width / 2))
    left_eye_rect = (left_eye_rect_point1, left_eye_rect_point2)
    cells['left_eye'] = left_eye_rect

    key = 'right_eye'
    point = lmks[key][3]
    right_eye_rect_point1 = (point[0], int(point[1] - cell_width / 2))
    right_eye_rect_point2 = (point[0] + cell_width, int(point[1] + cell_width / 2))
    right_eye_rect = (right_eye_rect_point1, right_eye_rect_point2)
    cells['right_eye'] = right_eye_rect

    left_point = lmks['left_eyebrow'][2]
    right_point = lmks['right_eyebrow'][2]
    center_point = (int((left_point[0] + right_point[0]) / 2),
                    int((left_point[1] + right_point[1]) / 2))

    center_eyebrow_rect = get_rect(center_point, cell_width)
    cells['center_eyebrow'] = center_eyebrow_rect

    left_rect_point1 = (int(center_point[0] - cell_width * 3 / 2),
                        int(center_point[1] - cell_width / 2))
    left_rect_point2 = (int(center_point[0] - cell_width * 1 / 2),
                        int(center_point[1] + cell_width / 2))
    left_eyebrow_rect = (left_rect_point1, left_rect_point2)
    cells['left_eyebrow'] = left_eyebrow_rect

    right_rect_point1 = (int(center_point[0] + cell_width * 1 / 2),
                         int(center_point[1] - cell_width / 2))
    right_rect_point2 = (int(center_point[0] + cell_width * 3 / 2),
                         int(center_point[1] + cell_width / 2))
    right_eyebrow_rect = (right_rect_point1, right_rect_point2)
    cells['right_eyebrow'] = right_eyebrow_rect

    return cells, cell_width

# 输入参数为细胞在当前时刻t的值cell_t,起始时刻onset的值cell_onset,结束时刻offset的值cell_offset,以及一个衰减常数cell_epsilon
def compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon):

    numerator = (np.abs(cell_t - cell_onset) + 1.0)   # np.abs对两者差值取绝对值,避免正负抵消
    denominator = (np.abs(cell_t - cell_epsilon) + 1.0)
    difference = numerator / denominator

    numerator = (np.abs(cell_t - cell_offset) + 1.0)
    difference1 = numerator / denominator

    difference = difference + difference1

    return difference.mean()


def compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon):

    lmks = detect_lmks(frame_t)  # 检测当前帧的关键点
    cell_locations, cell_width = get_cell_locations(lmks)
    cell_differences = {}
    frame_t = frame_t.astype(np.float32)
    on_frame = on_frame.astype(np.float32)
    off_frame = off_frame.astype(np.float32)
    frame_epsilon = frame_epsilon.astype(np.float32)

    for key in cell_locations:
        cell_location = cell_locations[key]
        cell_t = get_cell(frame_t, cell_location)
        cell_onset = get_cell(on_frame, cell_location)
        cell_offset = get_cell(off_frame, cell_location)
        cell_epsilon = get_cell(frame_epsilon, cell_location)

        cell_difference = compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon)
        cell_differences[key] = cell_difference
    return cell_differences

# 在一个视频片段的所有帧上,提取每帧的特征,并找到特征峰值最大的帧,作为顶点帧
def find_apex_frame_of_clip(frame_paths):
    epsilon = 1

    on_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    off_frame = cv2.imread(frame_paths[-1], cv2.IMREAD_GRAYSCALE)

    features = []

    for i in range(epsilon, len(frame_paths)):
        frame_t = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
        frame_epsilon = cv2.imread(frame_paths[i - epsilon], cv2.IMREAD_GRAYSCALE)
        current_features = compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon)
        feature = 0
        for key in current_features:
            feature += current_features[key]
        feature = feature / len(current_features)
        features.append(feature)

    padding = [0.0] * epsilon
    features = np.array(padding + features)
    apex_frame_idx = features.argmax()
    apex_frame_path = frame_paths[apex_frame_idx]

    return apex_frame_path, features, apex_frame_idx


def draw_avg_plot(features, pred_apex_idx, data, clip_name):
    x = list(range(len(features)))
    plt.plot(x, features)
    plt.axvline(x=pred_apex_idx, label='pred apex idx at={}'.format(pred_apex_idx), c='red')
    plt.legend()
    plt.savefig('plots/{}/{}.png'.format(data, clip_name))
    plt.clf()
    plt.cla()
    plt.close()


def on_all_smic_clips():
    smic = pd.read_excel(smic_annotation_file)
    labels = []
    apex_frame_indices = []
    on_frame_paths = []
    off_frame_paths = []
    apex_frame_paths = []
    samples = zip(list(smic['Subject']),
                  list(smic['Filename']),
                  list(smic['OnsetF']),
                  list(smic['OffsetF']),
                  list(smic['Emotion']))

    with tqdm(total=164) as progress_bar:
        for subject, filename, on_frame_idx, off_frame_idx, emotion in samples:
            # Get all ME paths of a clip
            # 得到了所有的频帧的地址视，都存到了clip_frame_paths这个列表当中
            clip_frame_paths = get_clip_frame_paths(subject, filename, on_frame_idx, off_frame_idx)

            # Find apex frame paths   找到峰值帧的地址
            apex_frame_path, features, apex_relative_idx = find_apex_frame_of_clip(clip_frame_paths)
            # draw_avg_plot(features, apex_relative_idx, 'smic', filename)

            on_frame_paths.append(clip_frame_paths[0])
            off_frame_paths.append(clip_frame_paths[-1])
            apex_frame_paths.append(apex_frame_path)

            print(apex_frame_path)
            apex_frame_idx = int(apex_frame_path.split('\\')[-1].split('.')[0].replace('image', ''))
            apex_frame_indices.append(apex_frame_idx)

            # Label
            labels.append(label_dict[emotion])
            progress_bar.update(1)


    # Save data_to_csv file
    data_dict = {'data'            : ['smic'] * len(labels),
                 'subject'         : list(smic['Subject']),
                 'clip'            : list(smic['Filename']),
                 'label'           : labels,
                 'onset_frame'     : list(smic['OnsetF']),
                 'apex_frame'      : apex_frame_indices,
                 'offset_frame'    : list(smic['OffsetF']),
                 'onset_frame_path': on_frame_paths,
                 'apex_frame_path' : apex_frame_paths,
                 'off_frame_path'  : off_frame_paths}
    smic_data = pd.DataFrame.from_dict(data_dict)
    smic_data.to_csv('SMIC_apex_all_new.csv', header=True, index=None)


if __name__ == '__main__':

    on_all_smic_clips()