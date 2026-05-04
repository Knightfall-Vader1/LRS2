import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for license plates")
    parser.add_argument("--labels", default="dataset/ocr_labels.csv")
    parser.add_argument("--images", default="dataset/ocr_crops",
                        help="Directory containing plate crop images")
    parser.add_argument("--output", default="models/trocr")
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    for pkg in ("transformers", "torch", "PIL"):
        mod = "PIL" if pkg == "PIL" else pkg
        if importlib.util.find_spec(mod) is None:
            raise SystemExit(f"{pkg} is not installed. pip install {pkg if pkg != 'PIL' else 'Pillow'}")

    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"Labels file not found at {labels_path}")

    images_dir = Path(args.images)
    if not images_dir.exists():
        raise SystemExit(
            f"Images directory not found at {images_dir}. "
            "Run scripts/generate_ocr_crops.py first to create plate crops."
        )

    df = pd.read_csv(labels_path)
    if "File" not in df.columns or "Plate text" not in df.columns:
        raise SystemExit("Labels CSV must include 'File' and 'Plate text' columns")

    # Filter to rows whose image file actually exists
    df = df[df["File"].apply(lambda f: (images_dir / f).exists())].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(
            f"No matching images found in {images_dir} for the entries in {labels_path}. "
            "Check that 'File' values in the CSV match filenames in the images directory."
        )

    print(f"Training on {len(df)} labeled plate crops.")

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)

    # Required decoder config for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    class PlateOCRDataset(Dataset):
        def __init__(self, dataframe: pd.DataFrame, img_dir: Path, proc: TrOCRProcessor) -> None:
            self.df = dataframe
            self.img_dir = img_dir
            self.proc = proc

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            image = Image.open(self.img_dir / row["File"]).convert("RGB")
            pixel_values = self.proc(images=image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.proc.tokenizer(
                row["Plate text"],
                padding="max_length",
                max_length=32,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            # Replace padding token id with -100 so it is ignored in the loss
            labels[labels == self.proc.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    dataset = PlateOCRDataset(df, images_dir, processor)

    # Simple 80/20 train/val split
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    main()
