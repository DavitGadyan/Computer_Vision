"""Convert a Roboflow-style COCO export (split/_annotations.coco.json) into an Ultralytics YOLO dataset."""
import argparse
import json
import shutil
from pathlib import Path

import yaml


def convert_split(src: Path, dst: Path, split: str, cat_map: dict) -> int:
    ann = json.loads((src / "_annotations.coco.json").read_text())
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for img in ann["images"]:
        w, h = img["width"], img["height"]
        lines = []
        for a in by_img.get(img["id"], []):
            if a["category_id"] not in cat_map:
                continue
            x, y, bw, bh = a["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            lines.append(f"{cat_map[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        shutil.copy2(src / img["file_name"], dst / "images" / split / img["file_name"])
        (dst / "labels" / split / Path(img["file_name"]).with_suffix(".txt").name).write_text("\n".join(lines))
    return len(ann["images"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train/ valid/ test/ COCO splits")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--skip-classes", nargs="*", default=[], help="category names to drop (e.g. Roboflow super-category)")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    cats = json.loads((src / "train" / "_annotations.coco.json").read_text())["categories"]
    keep = [c for c in cats if c["name"] not in args.skip_classes]
    cat_map = {c["id"]: i for i, c in enumerate(keep)}
    for split, out in [("train", "train"), ("valid", "val"), ("test", "test")]:
        if (src / split).exists():
            print(split, convert_split(src / split, dst, out, cat_map))
    data = {"path": str(dst.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: c["name"] for i, c in enumerate(keep)}}
    (dst / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print("wrote", dst / "data.yaml", data["names"])


if __name__ == "__main__":
    main()
