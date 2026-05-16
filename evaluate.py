import os
from ultralytics import YOLO


import ultralytics.utils.downloads as dl
if hasattr(dl, '_patched_get'):
    delattr(dl, '_patched_get')
    print("已移除网络禁用补丁，允许必要下载")

os.environ.pop('ULTRALYTICS_DISABLE_NETWORK', None)


MODEL_PATH = r"C:\Users\wyx12\PyCharmMiscProject\proect\runs\detect\runs\detect\bottle_liquid\weights\best.pt"

DATA = 'data.yaml'
CONF = 0.25
IOU = 0.45
# ==========================

def main():
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        print("请确认训练已成功保存模型，或将 MODEL_PATH 改为正确的文件路径")
        return

    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 在验证集上评估
    metrics = model.val(
        data=DATA,
        split='val',
        imgsz=640,
        conf=CONF,
        iou=IOU
    )

    # 输出指标
    print('=' * 50)
    print('模型评估结果')
    print('=' * 50)
    print(f'mAP@0.5:       {metrics.box.map50:.4f}')
    print(f'mAP@0.5:0.95:  {metrics.box.map:.4f}')
    print(f'Precision:     {metrics.box.mp:.4f}')
    print(f'Recall:        {metrics.box.mr:.4f}')

    # 各类别详细指标
    if hasattr(metrics.box, 'ap_class_index'):
        ap = metrics.box.ap
        class_index = metrics.box.ap_class_index
        names = model.names
        print('\n各类别AP:')
        for idx, ap_value in zip(class_index, ap):
            print(f'  {names[idx]}: {ap_value:.4f}')

if __name__ == '__main__':
    main()