from ultralytics import YOLO

model = YOLO(r"runs\segment\runs\foodseg103_finetune\weights\best.pt")

results = model.predict(
    source=r"datasets\foodseg103_yolo\images\val\00000048.jpg",
    conf=0.001,
    verbose=False
)

r = results[0]

print("Boxes:", len(r.boxes))

for box in r.boxes:
    cls = int(box.cls.item())
    conf = float(box.conf.item())

    print(f"{model.names[cls]:25} {conf:.4f}")