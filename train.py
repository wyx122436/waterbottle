from ultralytics import YOLO

MODEL = 'yolov8s.pt'
DATA = 'data.yaml'
EPOCHS = 100
IMG_SIZE = 320
BATCH = 8
FREEZE = 10
OPTIMIZER = 'AdamW'
LR0 = 0.001
DEVICE = 'cpu'
workers=4
cache=True

def main():
    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        freeze=FREEZE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        cos_lr=True,
        device=DEVICE,
        project='runs/detect',
        name='bottle_liquid',
        exist_ok=True,
        amp=True
    )
    print('训练完成！')

if __name__ == '__main__':
    main()