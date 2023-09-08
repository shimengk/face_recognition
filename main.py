import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog, \
    QMessageBox, QApplication

import common
import my_utils
from face_recognition_pre import FaceRecognition
from model.DBFace import DBFace

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")


# 人脸对齐
def transform_face(image, landmarks, landmarks_history=[], output_size=192, smoothing_window=5):
    # smoothing_window图片时为1，视频为5
    # 将当前关键点添加到历史记录
    landmarks_history.append(np.array(landmarks).reshape([-1, 2]))
    # 如果历史记录大于平滑窗口的带下，则删除最早的点
    if len(landmarks_history) > smoothing_window:
        landmarks_history.pop(0)
    # 计算平滑后的关键点
    smoothed_marks = np.mean(landmarks_history, axis=0)

    # 变换后的五官坐标
    key_marks = np.array([65, 65, 130, 65, 96, 96, 75, 130, 117, 130]).reshape([-1, 2])
    # 获取变换矩阵
    M, _ = cv2.estimateAffinePartial2D(smoothed_marks, key_marks)
    # 执行变换操作
    transformed = cv2.warpAffine(image, M, (output_size, output_size), borderValue=0.0)

    offset = 30
    transformed = transformed[0 + offset:192 - offset, 0 + offset:192 - offset]
    return transformed


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def detect_image(model, file):
    image = common.imread(file)
    objs = detect(model, image)

    for obj in objs:
        common.drawbbox(image, obj)

    common.imwrite("detect_result/" + common.file_name_no_suffix(file) + ".draw.jpg", image)


def detect_image_align_face(model, file, recognition):
    image = common.imread(file)
    objs = detect(model, image)

    for obj in objs:
        transformed = transform_face(image, obj.landmark, smoothing_window=5)
        # 展示结果
        cv2.imshow("transformed", transformed)
        name, sim = recognition(transformed)

        threshold = 0.3
        common.drawbbox(image, obj, name, sim, threshold)
        cv2.imshow("transformed", transformed)
        # common.drawbbox(image, obj)
        # cv2.imshow("src", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def image_demo():
    # dbface = DBFace()
    # dbface.eval()
    #
    # if HAS_CUDA:
    #     dbface.cuda()
    #
    # dbface.load("model/dbface.pth")
    #
    # for file in os.listdir():
    #     # detect_image(dbface, f"datas/{file}")
    #     detect_image_align_face(dbface, f"datas2/{file}")

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    recognition = FaceRecognition()

    root = r'D:\data\face_group'
    for name in os.listdir(root)[2:]:
        dir_name = os.path.join(root, name)
        for file in os.listdir(dir_name):
            # detect_image(dbface, f"datas/{file}")
            detect_image_align_face(dbface, os.path.join(dir_name, file), recognition)


def camera_recognition(video_path):
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    recognition = FaceRecognition()
    frame_counter = 0

    while ok:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        objs = detect(dbface, frame)
        # # 每120帧刷新人脸识别结果
        # if frame_counter % 10 == 0 or frame_counter == 0:
        #     res_dict = {}
        #     cur_name = 'recing'
        #     cur_max_times = 0

        for obj in objs:
            transformed = transform_face(frame, obj.landmark, smoothing_window=5)
            # 展示结果
            cv2.imshow("transformed", cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
            name, max_times = recognition(transformed)
            #
            # if name not in res_dict.keys():
            #     res_dict[name] = max_times
            # else:
            #     res_dict[name] += max_times
            # # 最大频次占比的阈值，大于这个频次则判断是这个人
            time_threshold = 0.9
            # sorted_dict = dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True))
            # if frame_counter % 20 == 0 and frame_counter > 20:
            #     cur_name = next(iter(sorted_dict))  # 获取第一个键
            #     cur_max_times = sorted_dict[name]

            common.drawbbox_rec(frame, obj, name, max_times, time_threshold)

        cv2.imshow("demo DBFace", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


def face_direction(landmark):
    now_pos = None
    left_eye, right_eye, nose, left_mouth, right_mouth = landmark
    xc, yc = (left_eye[0] + right_eye[0]) / 2, (left_eye[1] + left_mouth[1]) / 2

    f = 50  # 人脸距离相机距离
    # 计算俯仰角  判断上下
    dy = yc - nose[1]
    pitch = np.arctan(dy / f)
    pitch = pitch * 180 / np.pi
    # 计算偏航角  判断左右
    dx = nose[0] - xc
    yaw = np.arctan(dx / f)
    yaw = yaw * 180 / np.pi

    print(f'yaw: {round(yaw, 2)}, pitch: {round(yaw, 2)}')
    abs_yaw, abs_pitch = abs(yaw), abs(pitch)

    if abs_pitch < 5:
        now_pos = 'center' if abs_yaw < 5 else 'right' if yaw > 5 else 'left'
    elif pitch > 5:
        now_pos = 'top' if abs_yaw < 5 else "right_top" if yaw > 5 else "left_top"
    else:
        now_pos = 'bottom' if abs_yaw < 5 else "right_bottom" if yaw > 5 else "left_bottom"
    return now_pos


def camera_register(video_path, name):
    dbface = DBFace()
    dbface.eval()
    recognition = FaceRecognition()
    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    cap = cv2.VideoCapture(video_path)
    W = 640
    H = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, frame = cap.read()

    # 人脸注册，脸需要放进设定框
    CENTER = (W // 2, H // 2)
    RADIUS = H // 3
    COLOR_HAS = (0, 255, 0)
    COLOR_NOT = (0, 0, 255)
    REGULAR_BOX = (CENTER[0] - RADIUS // 2,
                   CENTER[1] - RADIUS // 2,
                   CENTER[0] + RADIUS // 2,
                   CENTER[1] + RADIUS // 2)

    iou_threshold = 0.3

    # 各方向的中心点，用于画出当前的注意力点
    POS_DICT = {
        'center': CENTER,
        'left': (CENTER[0] - RADIUS, CENTER[1]),
        'right': (CENTER[0] + RADIUS, CENTER[1]),
        'top': (CENTER[0], CENTER[1] - RADIUS),
        'bottom': (CENTER[0], CENTER[1] + RADIUS),
        "left_top": (CENTER[0] - RADIUS // np.sqrt(2), CENTER[1] - RADIUS // np.sqrt(2)),
        "right_top": (CENTER[0] + RADIUS // np.sqrt(2), CENTER[1] - RADIUS // np.sqrt(2)),
        "left_bottom": (CENTER[0] - RADIUS // np.sqrt(2), CENTER[1] + RADIUS // np.sqrt(2)),
        "right_bottom": (CENTER[0] + RADIUS // np.sqrt(2), CENTER[1] + RADIUS // np.sqrt(2))
    }
    # 是否检测到该位置人脸
    pos_counter = {
        'center': 0,
        'left': 0,
        'right': 0,
        'top': 0,
        'bottom': 0,
        "left_top": 0,
        "right_top": 0,
        "left_bottom": 0,
        "right_bottom": 0
    }
    # 每个位置对应的画圈角度
    ANGLE_DICT = {
        'center': (0, 0),
        'left': (90, 135),
        'right': (270, 315),
        'top': (180, 225),
        'bottom': (0, 45),
        "left_top": (135, 180),
        "right_top": (225, 270),
        "left_bottom": (45, 90),
        "right_bottom": (-45, 0)
    }
    frame_counter = 0
    # 每个方向截取的图片数
    MAX_PIC_NUM = 10
    save_counter = 0
    last_center_point = (0, 0)
    feature_list = []
    # ===============================================================================================================
    while ok:
        objs = detect(dbface, frame)
        text = ''
        circle_color = COLOR_NOT
        now_pos = None
        for obj in objs:

            # 检测到的人脸框，与设定框iou判断是否在范围内
            detect_box = obj.box

            iou = my_utils.IOU(REGULAR_BOX, detect_box)
            # print(iou)
            if iou > iou_threshold:
                circle_color = COLOR_HAS
                text = 'writing'
                detect_center = obj.landmark[2]

                """
                计算欧拉角判断人脸方向，偏航角判断左右，俯仰角判断上下
                left_eye, right_eye, nose, left_mouth, right_mouth = keypoints
                """
                now_pos = face_direction(obj.landmark)

                # 如果是这几个位置，且没有超过最大截取图片数,且与上次的点有一定距离。截取保存
                if now_pos \
                        and pos_counter[now_pos] < MAX_PIC_NUM \
                        and my_utils.distance(last_center_point, detect_center) > 3:
                    pos_counter[now_pos] += 1
                    # 正脸对齐，其他不用
                    is_align = True if now_pos == 'center' else False

                    feature = crop_and_save(frame, detect_box, is_align, obj.landmark, save_counter, name, recognition)
                    feature_list.append(feature)
                    save_counter += 1
                last_center_point = detect_center

            else:
                circle_color = COLOR_NOT
                text = 'no face'

            common.drawbbox(frame, obj)
        # ===============================================================================================================
        # 除人脸框外进行高斯模糊。 创建遮罩，中心圆区域为1，其他区域为0
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, (CENTER[0], CENTER[1]), RADIUS + 4, 1, -1)
        # 定义中心圆的半径和高斯模糊半径
        circle_radius = 33
        gaussian_blur_radius = 33
        # 对图像进行高斯模糊
        blurred_image = cv2.GaussianBlur(frame, (gaussian_blur_radius, gaussian_blur_radius), 0)
        # 仅保留中心圆部分
        output_image = frame.copy()
        output_image[mask == 0] = blurred_image[mask == 0]

        # 画固定框和文字
        cv2.putText(output_image, text, (20, 20), 0, 1, (255, 0, 0), 1, 16)
        cv2.circle(output_image, CENTER, RADIUS, circle_color, thickness=2)
        cv2.circle(output_image, CENTER, 2, circle_color, thickness=-1)
        # 画当前注意力点
        if now_pos:
            cv2.circle(output_image, intv(POS_DICT[now_pos]), 6, (0, 255, 255), thickness=-1)
        # 画检测结果，如果当前方向检测完成，标记
        for key in POS_DICT.keys():
            pos = intv(POS_DICT[key])
            cv2.putText(output_image, str(pos_counter[key]), pos, 0, 0.5, (255, 0, 0), 1, 16)
            if pos_counter[key] == MAX_PIC_NUM:
                start_angle, end_angle = ANGLE_DICT[key]
                cv2.ellipse(output_image, CENTER, (RADIUS, RADIUS), 45, start_angle + 22.5, end_angle + 22.5,
                            (255, 0, 0), thickness=4)

        cv2.imshow("demo DBFace", output_image)
        # ===============================================================================================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 判断是否完成采集(每个方向都采集到MAX_PIC_NUM张图片)
        if sum(pos_counter.values()) == MAX_PIC_NUM * len(pos_counter.values()):
            # 保存特征文件
            stack_tensors = torch.stack(feature_list, dim=0).reshape(-1, 128)

            pt_path = f'res/feats/{name}.pt'
            os.makedirs(os.path.dirname(pt_path), exist_ok=True)
            torch.save(stack_tensors, pt_path)
            print('特征保存成功，注册成功')
            break
        else:
            ok, frame = cap.read()
            frame_counter += 1
    cap.release()
    cv2.destroyAllWindows()


def crop_and_save(img, box, is_align, landmark, save_counter, name, recognition):
    def save(save_img):
        save_path = f'res/imgs/{name}/{name}_{save_counter}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)

    if is_align:
        transformed = transform_face(img, landmark, smoothing_window=5)
        save(transformed)
        return recognition.get_feature(transformed)
    else:
        x, y, r, b = intv(box)
        w = r - x + 1
        h = b - y + 1
        if w > h:
            offset = (w - h) // 2
            crop_img = img[y:y + h, x + offset:x + w - offset]
        else:
            offset = (h - w) // 2
            crop_img = img[y + offset:y + h - offset, x:x + w]
        try:
            crop_img = cv2.resize(crop_img, (224, 224))
            save(crop_img)
            return recognition.get_feature(crop_img)
        except:
            print('save error')


def intv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


class Register:
    W = 640
    H = 480
    # 人脸注册，脸需要放进设定框
    CENTER = (W // 2, H // 2)
    RADIUS = H // 3
    COLOR_HAS = (0, 255, 0)
    COLOR_NOT = (255, 0, 0)
    REGULAR_BOX = (CENTER[0] - RADIUS // 2,
                   CENTER[1] - RADIUS // 2,
                   CENTER[0] + RADIUS // 2,
                   CENTER[1] + RADIUS // 2)

    iou_threshold = 0.2

    # 各方向的中心点，用于画出当前的注意力点
    POS_DICT = {
        'center': CENTER,
        'left': (CENTER[0] - RADIUS, CENTER[1]),
        'right': (CENTER[0] + RADIUS, CENTER[1]),
        'top': (CENTER[0], CENTER[1] - RADIUS),
        'bottom': (CENTER[0], CENTER[1] + RADIUS),
        "left_top": (CENTER[0] - RADIUS // np.sqrt(2), CENTER[1] - RADIUS // np.sqrt(2)),
        "right_top": (CENTER[0] + RADIUS // np.sqrt(2), CENTER[1] - RADIUS // np.sqrt(2)),
        "left_bottom": (CENTER[0] - RADIUS // np.sqrt(2), CENTER[1] + RADIUS // np.sqrt(2)),
        "right_bottom": (CENTER[0] + RADIUS // np.sqrt(2), CENTER[1] + RADIUS // np.sqrt(2))
    }

    # 每个位置对应的画圈角度
    ANGLE_DICT = {
        'center': (0, 0),
        'left': (90, 135),
        'right': (270, 315),
        'top': (180, 225),
        'bottom': (0, 45),
        "left_top": (135, 180),
        "right_top": (225, 270),
        "left_bottom": (45, 90),
        "right_bottom": (-45, 0)
    }
    # 每个方向截取的图片数
    MAX_PIC_NUM = 10

    def __init__(self, name):
        self.frame_counter = 0

        # 是否检测到该位置人脸
        self.pos_counter = {
            'center': 0,
            'left': 0,
            'right': 0,
            'top': 0,
            'bottom': 0,
            "left_top": 0,
            "right_top": 0,
            "left_bottom": 0,
            "right_bottom": 0
        }
        self.save_counter = 0
        self.last_center_point = (0, 0)
        self.feature_list = []
        self.name = name

    def __call__(self, dbface, frame, recognition):

        objs = detect(dbface, frame)
        text = ''
        circle_color = self.COLOR_NOT
        now_pos = None

        for obj in objs:

            # 检测到的人脸框，与设定框iou判断是否在范围内
            detect_box = obj.box

            iou = my_utils.IOU(self.REGULAR_BOX, detect_box)
            # print(iou)
            if iou < 0.1:
                continue
            elif iou > self.iou_threshold:
                circle_color = self.COLOR_HAS
                text = 'writing'
                detect_center = obj.landmark[2]

                """
                计算欧拉角判断人脸方向，偏航角判断左右，俯仰角判断上下
                left_eye, right_eye, nose, left_mouth, right_mouth = keypoints
                """
                now_pos = face_direction(obj.landmark)

                # 如果是这几个位置，且没有超过最大截取图片数,且与上次的点有一定距离。截取保存
                if now_pos \
                        and self.pos_counter[now_pos] < self.MAX_PIC_NUM \
                        and my_utils.distance(self.last_center_point, detect_center) > 3:
                    self.pos_counter[now_pos] += 1
                    # 正脸对齐，其他不用
                    is_align = True if now_pos == 'center' else False

                    feature = crop_and_save(frame, detect_box, is_align, obj.landmark, self.save_counter, self.name,
                                            recognition)
                    self.feature_list.append(feature)
                    self.save_counter += 1
                self.last_center_point = detect_center

            else:
                circle_color = self.COLOR_NOT
                text = 'no face'

            common.drawbbox(frame, obj)
        # ===============================================================================================================
        # 除人脸框外进行高斯模糊。 创建遮罩，中心圆区域为1，其他区域为0
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(mask, (self.CENTER[0], self.CENTER[1]), self.RADIUS + 4, 1, -1)
        # 定义中心圆的半径和高斯模糊半径
        circle_radius = 33
        gaussian_blur_radius = 33
        # 对图像进行高斯模糊
        blurred_image = cv2.GaussianBlur(frame, (gaussian_blur_radius, gaussian_blur_radius), 0)
        # 仅保留中心圆部分
        output_image = frame.copy()
        output_image[mask == 0] = blurred_image[mask == 0]

        # 画固定框和文字
        cv2.putText(output_image, text, (20, 20), 0, 1, (255, 0, 0), 1, 16)
        cv2.circle(output_image, self.CENTER, self.RADIUS, circle_color, thickness=2)
        cv2.circle(output_image, self.CENTER, 2, circle_color, thickness=-1)
        # 画当前注意力点
        if now_pos:
            cv2.circle(output_image, intv(self.POS_DICT[now_pos]), 6, (255, 255, 0), thickness=-1)
        # 画检测结果，如果当前方向检测完成，标记
        for key in self.POS_DICT.keys():
            pos = intv(self.POS_DICT[key])
            cv2.putText(output_image, str(self.pos_counter[key]), pos, 0, 0.5, (255, 0, 0), 1, 16)
            if self.pos_counter[key] == self.MAX_PIC_NUM:
                start_angle, end_angle = self.ANGLE_DICT[key]
                cv2.ellipse(output_image, self.CENTER, (self.RADIUS, self.RADIUS), 45, start_angle + 22.5,
                            end_angle + 22.5,
                            (0, 0, 255), thickness=4)

                # 判断是否完成采集(每个方向都采集到MAX_PIC_NUM张图片)
                if sum(self.pos_counter.values()) == self.MAX_PIC_NUM * len(self.pos_counter.values()):
                    # 保存特征文件
                    stack_tensors = torch.stack(self.feature_list, dim=0).reshape(-1, 128)

                    pt_path = f'res/feats/{self.name}.pt'
                    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
                    torch.save(stack_tensors, pt_path)
                    print('特征保存成功，注册成功')
                    return None

        return output_image


class OpenCVWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")

        self.video_capture = cv2.VideoCapture(0)
        self.video_frame = QLabel(self)

        self.register_button = QPushButton("人脸注册", self)
        self.register_button.clicked.connect(self.click_register)
        self.recognition_button = QPushButton("识别", self)
        self.recognition_button.clicked.connect(self.click_recognition)

        layout = QVBoxLayout()
        layout.addWidget(self.video_frame)
        layout.addWidget(self.register_button)
        layout.addWidget(self.recognition_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.timer.start(30)  # Update frame every 30 ms
        self.mode = 0  # 1为注册模式，2为登录模式
        # ===============================================================================================================
        device = 'cuda'
        self.dbface = DBFace().to(device)
        self.dbface.eval()
        self.dbface.load("model/dbface.pth")
        self.recognition = FaceRecognition()
        self.register = None
        self.input_name = None

    def click_register(self):
        text, ok = QInputDialog.getText(self, "人脸录入", "输入你的姓名:")
        if ok and text:
            self.input_name = text
            self.register = Register(self.input_name)
            self.mode = 1

    def click_recognition(self):
        self.mode = 2

    def show_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 录入模式
            if self.mode == 1:
                frame = self.register(self.dbface, frame, self.recognition)
                # 录入完成
                if frame is None:
                    self.show_message("录入成功", "你的人脸信息已录入系统")
                    self.mode = 0

            # 识别模式
            elif self.mode == 2:
                objs = detect(self.dbface, frame)
                for idx, obj in enumerate(objs):
                    transformed = transform_face(frame, obj.landmark, smoothing_window=5)
                    cv2.imshow(f"transformed{idx}", cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
                    name, max_times = self.recognition(transformed)
                    time_threshold = 0.02
                    common.drawbbox_rec(frame, obj, name, max_times, time_threshold)

            if frame is not None:
                height, width, channel = frame.shape
                image = QImage(frame.data, width, height, width * channel, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.video_frame.setPixmap(pixmap)


if __name__ == "__main__":

    mode = 1  # 1为识别人脸，2为注册人脸
    video_path = 0

    if mode == 1:
        camera_recognition(video_path)
    else:
        while True:
            name = input('请输入要录入的人名:')
            camera_register(video_path, name)
