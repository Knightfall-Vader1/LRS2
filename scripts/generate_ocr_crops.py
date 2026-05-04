"""
generate_ocr_crops.py

Runs YOLO plate detection on all images in the dataset and saves the highest-confidence
plate crop for each image to dataset/ocr_crops/.  Also writes a CSV template
(dataset/ocr_crops_template.csv) with a blank "Plate text" column so you can
fill in the ground-truth text manually.

Usage:
    python scripts/generate_ocr_crops.py \
        --model models/plate_detector.pt \
        --images dataset/images \
        --output dataset/ocr_crops \
        --conf 0.25
"""

import argparse
import csv
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-crop licence plates for TrOCR labeling")
    parser.add_argument("--model", default="models/plate_detector.pt",
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--images", default="dataset/images",
                        help="Root directory to search for images (searched recursively)")
    parser.add_argument("--output", default="dataset/ocr_crops",
                        help="Directory to save plate crop images")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Minimum detection confidence to accept a crop")
    parser.add_argument("--template", default="dataset/ocr_crops_template.csv",
                        help="Path for the output CSV labeling template")
    args = parser.parse_args()

    for pkg in ("ultralytics", "PIL"):
        mod = "PIL" if pkg == "PIL" else pkg
        if importlib.util.find_spec(mod) is None:
            raise SystemExit(
                f"{pkg} is not installed. pip install {'Pillow' if pkg == 'PIL' else pkg}"
            )

    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"YOLO weights not found at {model_path}")

    images_root = Path(args.images)
    if not images_root.exists():
        raise SystemExit(f"Images directory not found at {images_root}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        p for p in images_root.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        raise SystemExit(f"No images found under {images_root}")

    print(f"Found {len(image_paths)} images. Running detection...")

    rows: list[dict] = []
    saved = 0

    for img_path in sorted(image_paths):
        image = Image.open(img_path).convert("RGB")
        results = model.predict(source=np.array(image), verbose=False, conf=args.conf)

        # Collect all detections across result objects
        best_box = None
        best_conf = -1.0
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is None:
            continue  # no plate detected in this image

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        crop = image.crop((x1, y1, x2, y2))

        crop_filename = img_path.stem + "_crop.jpg"
        crop_path = output_dir / crop_filename
        crop.save(crop_path, format="JPEG", quality=95)

        rows.append({
            "File": crop_filename,
            "Plate text": "",
            "Country": "Germany",
            "Source image": str(img_path.relative_to(images_root) if img_path.is_relative_to(images_root) else img_path),
            "Confidence": f"{best_conf:.4f}",
        })
        saved += 1

    template_path = Path(args.template)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(template_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["File", "Plate text", "Country", "Source image", "Confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {saved} plate crops to {output_dir}/")
    print(f"Labeling template written to {template_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {template_path} and fill in the 'Plate text' column for each crop.")
    print(f"  2. Copy/append the completed rows into dataset/ocr_labels.csv.")
    print(f"  3. Run: python scripts/train_trocr.py --images {output_dir}")


if __name__ == "__main__":
    main()
