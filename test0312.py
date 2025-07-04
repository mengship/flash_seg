#!/user/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO


# 加载模型
model = YOLO("yolo11m-seg.pt")
# model = YOLO("last_l_seg_25000.pt")

# results=model('D:\\study_package\\ultralytics-main\\ultralytics\\assets\\bus.jpg')

# results[0].show()

def main():
    # Train the model
    train_results = model.train(
        # data="D:\\study_package\\ultralytics-main\\ultralytics\\cfg\\datasets\\coco128-seg.yaml",  # path to dataset YAML
        # data="D:\\study_package\\ultralytics-main\\ultralytics\\cfg\\datasets\\SKU-110K.yaml",
        data="D:\\python_Projects\\ultralytics-main\\ultralytics\\cfg\\datasets\\location-seg.yaml",
        epochs=10000,  # number of training epochs
        imgsz=640,  # training image size
        batch=16,
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        patience=0,
        # optimizer='Adam',  # 优化器选择
        # lr0=0.001,  # 初始学习率
    )
    # Evaluate model performance on the validation set
    metrics = model.val()

if __name__ == '__main__':
    main()