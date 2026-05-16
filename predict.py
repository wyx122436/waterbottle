from ultralytics import YOLO

MODEL_PATH = r"C:\Users\wyx12\PyCharmMiscProject\proect\runs\detect\runs\detect\bottle_liquid\weights\best.pt"
SOURCE = r"C:\Users\wyx12\PyCharmMiscProject\proect\dataset\valid\images"
CONF = 0.25
IOU = 0.45

def main():
    model = YOLO(MODEL_PATH)
    results = model.predict(source=SOURCE, conf=CONF, iou=IOU, imgsz=320, device='cpu', save=True)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box, conf, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                print(f'类别: {r.names[int(cls_id)]}, 置信度: {conf:.2f}')
        else:
            print('未检测到瓶子')

if __name__ == '__main__':
    main()