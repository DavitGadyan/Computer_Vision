"""CPU fine-tuning wrapper used by the custom-model projects (license plates, solar defects)."""
import argparse
import os
import shutil

from config import DEVICE, MODELS, RUNS, model_path  # noqa: E402  (sets CUDA_VISIBLE_DEVICES)
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True, help="run name; best weights are copied to models/<name>.pt")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set (faster CPU runs)")
    args = ap.parse_args()

    model = YOLO(model_path(args.model))
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=DEVICE,
                workers=min(16, os.cpu_count()), fraction=args.fraction, project=str(RUNS), name=args.name,
                exist_ok=True, patience=15, plots=True, cache="ram")
    best = RUNS / args.name / "weights" / "best.pt"
    shutil.copy2(best, MODELS / f"{args.name}.pt")
    metrics = YOLO(str(best)).val(data=args.data, split="test", device=DEVICE, project=str(RUNS), name=f"{args.name}_test", exist_ok=True)
    print(f"TEST mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
