# 👗 Multimodal Fashion Classifier

Classifying fashion products using **both images and text** — because a picture of a shirt and the words "men's casual shirt" together tell you more than either one alone.

Built with ResNet-18 + DistilBERT, fused through a custom gated cross-attention mechanism.

---

## What it does

Takes a product image and its text description (name, article type, colour), runs them through two separate encoders, fuses the representations, and predicts one of 10 fashion categories.

```
Image  →  ResNet-18  →  512-dim vector  ↘
                                          GatedFusion  →  MLP  →  category
Text   →  DistilBERT →  512-dim vector  ↗
```

---

## Why multimodal?

A plain text model struggles when product names are vague ("Basic Tee"). A plain image model struggles when products look similar ("Oxford Shirt" vs "Casual Shirt"). Using both modalities together significantly reduces ambiguity.

---

## Architecture

| Component | Details |
|---|---|
| Image Encoder | ResNet-18 pretrained on ImageNet. Layers 1–2 frozen, layers 3–4 fine-tuned. Output projected to 512-dim. |
| Text Encoder | DistilBERT. Layers 0–4 frozen, last transformer layer fine-tuned. CLS token → 512-dim projection. |
| Fusion | GatedFusion: element-wise gating + multi-head cross-attention (4 heads) + residual norm |
| Classifier | Linear(1024→512) → BN → GELU → Dropout → Linear(512→256) → Linear(256→N) |

---

## Dataset

[Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) — ~44K products across 10 master categories.

Split: **70% train / 15% val / 15% test** (stratified).  
Class imbalance handled with `WeightedRandomSampler`.

---

## Training details

- **Optimizer:** AdamW with differential learning rates (pretrained layers get 10× smaller LR)
- **Scheduler:** CosineAnnealingLR
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Mixed precision:** `torch.cuda.amp` (GradScaler + autocast)
- **Early stopping:** patience = 5 epochs

---



---

## Quickstart

```bash
pip install -r requirements.txt
python data/download.py          # pulls dataset via kagglehub
python train.py
```

---

## Requirements

```
torch
torchvision
transformers
scikit-learn
pandas
Pillow
kagglehub
```

---

## Results

> Training in progress. Results will be updated after full run.

| Split | Accuracy | Macro F1 |
|---|---|---|
| Val   | —        | —        |
| Test  | —        | —        |

---

## Notes

- `NUM_WORKERS=0` required in Colab/Jupyter due to multiprocessing constraints
- Augmentation pipeline (RandomCrop, ColorJitter, RandomRotation) is implemented but currently disabled — can be re-enabled in `get_transforms()`
- Model checkpoint saved only when validation F1 improves (`best_model.pth`)
