from ultralytics import YOLO

def main():
    model = YOLO("yolov8s-seg.pt")

    model.train(
        data="training/vision/foodseg.yaml",

        epochs=100,
        patience=25,

        imgsz=640,
        batch=8,

        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=5e-4,

        cos_lr=True,

        amp=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10,
        translate=0.1,
        scale=0.5,

        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        workers=4,
        device=0,

        project="runs",
        name="foodseg103_final",

        save=True,
        plots=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()