#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a universal perturbation (UTAP-style) with CONCH as the image encoder.

- Adds a learnable delta in [0, 1] space (clamped to ±EPSILON_SCALED).
- Uses CONCH's eval transform for normalization.
- Minimizes cosine similarity between clean and attacked embeddings.
- Saves both the perturbation tensor (.pt) and a visualization (.png).

Requires:
    pip install conch-pytorch torchvision timm huggingface_hub pillow tqdm
"""

import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# --- CONCH Specific Imports ---
# If you installed conch via pip, this import should work.
# Repo: https://huggingface.co/MahmoodLab/conch
from conch.open_clip_custom import create_model_from_pretrained

# =========================
# Hyperparameters & Config
# =========================

# Directory for the training dataset (class-subfolder layout, e.g. ADI, BACK, ...)
TRAIN_DATA_DIR = "/scratch/hmaurya/PGD/NCT-CRC-HE-100K"   # <-- change to your training data root

# If you want to sample a fixed number per class (like your UNI2 code)
N_CLASSES = 9
N_IMAGES_PER_CLASS = 100
N_TOTAL_IMAGES = 900  # 9 classes * 100 images/class (informational)

# Image / loader
IMAGE_SIZE = 224   # CONCH default
L_EPOCHS = 10
B_BATCH_SIZE = 5
N_WORKERS = 4
PIN_MEMORY = True

# Perturbation budget (epsilon) and step size (alpha)
EPSILON_RAW = 20.0                # in pixel space [0..255]
EPSILON_SCALED = EPSILON_RAW / 255.0
ALPHA = 4.35e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output files
OUTPUT_PERTURB_TENSOR = "conch_utap_perturbation.pt"
OUTPUT_PERTURB_PNG    = "conch_utap_perturbation_visualization.png"

# ===================================
# Model: load CONCH + eval transforms
# ===================================

def get_conch_model_and_transforms(device):
    """
    Loads the CONCH image encoder and its evaluation transform.

    Returns:
        model (nn.Module): CONCH model (eval mode, params frozen)
        pre_norm_transform (transforms.Compose): Resize+Crop+ToTensor (NO normalization)
        normalize_transform (callable): The Normalize transform extracted from CONCH eval transform
    """
    print("Loading CONCH model and eval transform...")
    # 'hf_hub:MahmoodLab/conch' pulls the default weights
    model, conch_eval_transform = create_model_from_pretrained(
        'conch_ViT-B-16',
        checkpoint_path="hf_hub:MahmoodLab/conch"
    )
    model = model.to(device).eval()

    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False

    # The eval transform typically includes Resize, CenterCrop, ToTensor, and Normalize.
    # We want to:
    #   - Apply Resize/CenterCrop/ToTensor to build [0,1] images for adding delta
    #   - Then apply ONLY the final Normalize for model input
    # We'll separate these pieces from conch_eval_transform, assuming Normalize is last.
    if not hasattr(conch_eval_transform, "transforms"):
        raise RuntimeError("Unexpected CONCH eval transform structure.")

    eval_ops = conch_eval_transform.transforms
    if len(eval_ops) < 2:
        raise RuntimeError("CONCH eval transform seems incomplete.")

    # Extract Normalize as the last op, and build pre-norm ops from the rest
    if not isinstance(eval_ops[-1], transforms.Normalize):
        # Fall back: assume mean/std used by CLIP-like models if last isn't Normalize
        # (Should not happen for standard CONCH build, but safer than crashing.)
        print("Warning: last eval op is not Normalize; falling back to default CLIP mean/std.")
        normalize_transform = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        pre_ops = eval_ops
    else:
        normalize_transform = eval_ops[-1]
        pre_ops = eval_ops[:-1]

    # Ensure pre_ops ends with ToTensor (so we get [0,1] tensors)
    has_to_tensor = any(isinstance(op, transforms.ToTensor) for op in pre_ops)
    if not has_to_tensor:
        pre_ops = list(pre_ops) + [transforms.ToTensor()]
        print("Appended ToTensor() to pre-norm pipeline.")

    # Enforce image size to IMAGE_SIZE for consistency
    # (In case the CONCH eval transform uses different size in future)
    pre_ops_forced = []
    saw_resize_or_crop = False
    for op in pre_ops:
        if isinstance(op, (transforms.Resize, transforms.CenterCrop)):
            saw_resize_or_crop = True
            # Replace sizes with IMAGE_SIZE (keep behavior)
            if isinstance(op, transforms.Resize):
                pre_ops_forced.append(transforms.Resize(IMAGE_SIZE))
            elif isinstance(op, transforms.CenterCrop):
                pre_ops_forced.append(transforms.CenterCrop(IMAGE_SIZE))
        else:
            pre_ops_forced.append(op)
    if not saw_resize_or_crop:
        # If none found, add a standard resize+center-crop to IMAGE_SIZE
        pre_ops_forced = [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ] + pre_ops_forced

    pre_norm_transform = transforms.Compose(pre_ops_forced)

    print("CONCH model ready. Separated transforms:")
    print(" - pre_norm_transform:", pre_norm_transform)
    print(" - normalize_transform:", normalize_transform)
    return model, pre_norm_transform, normalize_transform

# ===============
# Custom Dataset
# ===============

class HistologyDataset(Dataset):
    """
    Dataset that loads up to N_IMAGES_PER_CLASS from each subfolder in TRAIN_DATA_DIR.
    Produces [0,1] tensors (pre-norm) so we can add delta in pixel space before normalization.
    """
    def __init__(self, data_dir, pre_norm_transform, n_images_per_class):
        self.pre_norm_transform = pre_norm_transform
        self.image_paths = []

        # Find class directories (ADI, BACK, DEB, etc.)
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_dirs.sort()
        print(f"Found {len(class_dirs)} classes: {class_dirs}")

        if len(class_dirs) != N_CLASSES:
            print(f"Warning: Expected {N_CLASSES} classes, but found {len(class_dirs)}.")

        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            all_images = []
            all_images.extend(glob.glob(os.path.join(class_path, "*.tif")))
            all_images.extend(glob.glob(os.path.join(class_path, "*.jpg")))
            all_images.extend(glob.glob(os.path.join(class_path, "*.png")))

            if len(all_images) >= n_images_per_class:
                sampled = random.sample(all_images, n_images_per_class)
            else:
                print(f"Warning: Class {class_dir} has only {len(all_images)} images. Using all.")
                sampled = all_images

            self.image_paths.extend(sampled)

        random.shuffle(self.image_paths)
        print(f"Created dataset with {len(self.image_paths)} total images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        img_01 = self.pre_norm_transform(img)   # -> [0,1], resized/cropped to IMAGE_SIZE
        return img_01

# ===========================
# Training (UTAP-style) Loop
# ===========================

def train_utap_with_conch():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # --- Model & transforms ---
    model, pre_norm_transform, normalize_transform = get_conch_model_and_transforms(device)

    # --- Data ---
    dataset = HistologyDataset(
        data_dir=TRAIN_DATA_DIR,
        pre_norm_transform=pre_norm_transform,
        n_images_per_class=N_IMAGES_PER_CLASS,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=B_BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # --- Universal perturbation (delta) in [0,1] space ---
    delta = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, requires_grad=True)

    # --- Loss ---
    # Minimize cosine similarity between clean and attacked embeddings
    cosine_sim = nn.CosineSimilarity(dim=1)

    print("\n--- Starting UTAP (CONCH) ---")
    print(f"Epochs: {L_EPOCHS}, Batch Size: {B_BATCH_SIZE}")
    print(f"Epsilon (raw): {EPSILON_RAW}, Epsilon (scaled): {EPSILON_SCALED:.6f}")
    print(f"Alpha (step size): {ALPHA:.2e}")

    for epoch in range(L_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{L_EPOCHS}")
        running = 0.0
        pbar = tqdm(dataloader)

        for images_01 in pbar:
            images_01 = images_01.to(device)                   # [B,3,H,W] in [0,1]

            # Attacked images in [0,1]
            attacked_01 = torch.clamp(images_01 + delta, 0.0, 1.0)

            # Normalize both clean and attacked for CONCH
            clean_in  = normalize_transform(images_01)          # ready for encode_image
            attack_in = normalize_transform(attacked_01)

            # Extract features
            with torch.no_grad():
                clean_feat = model.encode_image(clean_in)       # shape [B, D]
                # If your model requires normalize=True, use: model.encode_image(clean_in, normalize=True)

            attack_feat = model.encode_image(attack_in)         # shape [B, D]
            # If needed: attack_feat = model.encode_image(attack_in, normalize=True)

            # Cosine similarity (we minimize it)
            loss = cosine_sim(clean_feat, attack_feat).mean()

            # Backprop ONLY through delta
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            with torch.no_grad():
                # PGD-style step in sign(grad) direction to minimize similarity
                grad = delta.grad.data
                delta.data = delta.data - ALPHA * torch.sign(grad)
                # Project back to L_inf ball
                delta.data = torch.clamp(delta.data, -EPSILON_SCALED, EPSILON_SCALED)

            running += loss.item()
            pbar.set_description(f"Batch Loss: {loss.item():.4f}")

        avg = running / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} Average Cosine Similarity (Loss): {avg:.6f}")

    print("\n--- Training Complete ---")

    # Save perturbation tensor
    torch.save(delta.detach().cpu(), OUTPUT_PERTURB_TENSOR)
    print(f"Saved perturbation tensor to: {OUTPUT_PERTURB_TENSOR}")

    # Save visualization: map [-ε, +ε] -> [0,1]
    vis = (delta.detach().cpu() + EPSILON_SCALED) / (2 * EPSILON_SCALED)
    save_image(vis, OUTPUT_PERTURB_PNG)
    print(f"Saved perturbation visualization to: {OUTPUT_PERTURB_PNG}")


if __name__ == "__main__":
    train_utap_with_conch()
