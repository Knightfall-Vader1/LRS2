# LRS2.0 (License Recognition System)

Backend for German license plate detection (YOLO) and recognition (TrOCR) with an authorization whitelist.

## Current Dataset Notes
- Pipeline test: 10 images (8 train / 2 val).
- YOLO training target: 1k+ images.
- TrOCR fine-tuning target: 10k+ plate crops with text labels.

## OCR Normalization Rules (German Plates)
- Uppercase letters only, single spaces between blocks.
- Only characters A–Z and 0–9.
- No hyphens or punctuation.
- AE/OE/UE are treated as normal letter sequences.
- No automatic substitution of ambiguous characters (O/0, B/8, I/1).

## API Overview (FastAPI)
- `POST /infer/image`: Upload an image for detection + recognition.
- `POST /infer/stream`: Placeholder for camera streaming endpoint.
- `GET /authorized`: List authorized plates.
- `POST /authorized`: Add a plate to the whitelist.
- `DELETE /authorized/{plate_text}`: Remove a plate from the whitelist.

## Quickstart
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Example Request
```bash
curl -F "file=@dataset/images/val/27716775.jpg" http://localhost:8000/infer/image
```

### Windows PowerShell Notes
PowerShell aliases `curl` to `Invoke-WebRequest`, which does not support `-F`. Use one of the options below in a **new terminal** while Uvicorn is running:

```powershell
# Option 1: Call the real curl executable
curl.exe -F "file=@dataset/images/val/26575120.jpg" http://localhost:8000/infer/image
```

```powershell
# Option 2: Use Invoke-WebRequest with multipart form data
$form = @{ file = Get-Item "dataset\\images\\val\\27716775.jpg" }
Invoke-WebRequest -Uri "http://localhost:8000/infer/image" -Method Post -Form $form
```

## Troubleshooting
### `Form data requires "python-multipart" to be installed`
The image upload endpoint requires `python-multipart`. Install dependencies
after pulling the latest changes:
```bash
pip install -r requirements.txt
```

## Training
### YOLO (Plate Detection)
```bash
python scripts/train_yolo.py --data dataset/data.yaml --epochs 100 --img 640
```

### TrOCR (Plate Recognition)
```bash
python scripts/train_trocr.py --labels dataset/ocr_labels.csv --output models/trocr
```

## Authorized Plates Store
Authorized plates live in `authorized_plates.json` in this format:
```json
{
  "plates": ["B AB 123", "M X 77"]
}
```

