import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 plate detector")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (-1 = auto)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--mosaic", type=float, default=0.8,
                        help="Mosaic augmentation probability (0.0–1.0)")
    parser.add_argument("--device", default="cpu",
                        help="Training device: 'cpu', 'cuda', '0', etc.")
    args = parser.parse_args()

    if importlib.util.find_spec("ultralytics") is None:
        raise SystemExit("ultralytics is not installed. pip install ultralytics")

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found at {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.img,
        patience=args.patience,
        batch=args.batch,
        lr0=args.lr0,
        mosaic=args.mosaic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
