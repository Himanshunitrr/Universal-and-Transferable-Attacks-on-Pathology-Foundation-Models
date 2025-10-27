import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import timm
from huggingface_hub import login, hf_hub_download
from PIL import Image
from tqdm import tqdm
import numpy as np

# --- Plotting Imports ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Please install matplotlib and seaborn to save plots: pip install matplotlib seaborn")
    plt = None
    sns = None

# --- Configuration ---
VAL_DATA_DIR = "/scratch/hmaurya/PGD/CRC-VAL-HE-7K"
PERTURBATION_PATH = "utap_perturbation.pt"
PROBE_PATH = "linear_probe_1epoch.pt" # Path to load the trained probe
SAMPLES_SAVE_DIR = "attacked_samples_fft_defended" # Directory to save sample defended images

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
FEATURE_DIM = 1536
BATCH_SIZE = 64 # Adjust based on your VRAM
NUM_SAMPLES_TO_SAVE = 5
FFT_RADIUS = 50 # Set the defense radius as requested

# --- 1. FFT Defense Functions ---

def create_circular_mask(h, w, center=None, radius=None):
    """ Creates a 2D circular low-pass filter mask. """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = torch.sqrt(
        (X.float() - center[0])**2 + (Y.float() - center[1])**2
    )
    mask = (dist_from_center <= radius).float()
    return mask.to(DEVICE)

def apply_fft_filter_batch(image_batch, radius):
    """
    Applies a low-pass filter to a BATCH of 3-channel image tensors.
    """
    B, C, H, W = image_batch.shape
    
    # 1. Create the circular mask (same mask for all in batch)
    mask = create_circular_mask(H, W, radius=radius) # [H, W]
    mask_batch = mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W], ready for broadcast

    # 2. Apply FFT and shift
    # torch.fft.fft2 operates on the last 2 dims, so [B, C, H, W] -> [B, C, H, W]
    f_transform = torch.fft.fft2(image_batch)
    f_transform_shifted = torch.fft.fftshift(f_transform, dim=(-2, -1))
    
    # 3. Apply the low-pass filter (mask)
    f_transform_filtered = f_transform_shifted * mask_batch
    
    # 4. Unshift and apply Inverse FFT
    f_ishift = torch.fft.ifftshift(f_transform_filtered, dim=(-2, -1))
    img_filtered_batch = torch.fft.ifft2(f_ishift)
    
    # 5. Get the real part and clamp
    img_filtered_batch = torch.real(img_filtered_batch)
    img_filtered_batch = torch.clamp(img_filtered_batch, 0.0, 1.0)
    
    return img_filtered_batch

# --- 2. Model & Data Setup ---

def get_model(device):
    """ Downloads and initializes the frozen UNI2-h model. """
    print("Loading UNI2-h model...")
    local_dir = "./assets/ckpts/uni2-h/"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        print("Downloading UNI2-h model...")
        hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    else:
        print("UNI2-h model already downloaded.")

    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224', 'img_size': IMAGE_SIZE,
        'patch_size': 14, 'depth': 24, 'num_heads': 24, 'init_values': 1e-5,
        'embed_dim': FEATURE_DIM, 'mlp_ratio': 2.66667*2, 'num_classes': 0,
        'no_embed_class': True, 'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model(**timm_kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("UNI2-h model loaded and frozen.")
    return model

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.layer(x)

class HistologyProbeDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.samples = []
        class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}
        self.classes = class_dirs
        print(f"Found {len(self.classes)} classes in {data_dir}: {self.class_to_idx}")
        for class_name, label_idx in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            for img_file in glob.glob(os.path.join(class_path, "*.tif")):
                self.samples.append((img_file, label_idx))
            for img_file in glob.glob(os.path.join(class_path, "*.jpg")):
                self.samples.append((img_file, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, label, img_path
        except Exception as e:
            print(f"Warning: Error loading {img_path}, skipping. Error: {e}")
            return self.transform(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))), 0, "error_image.png"

transform_pre_norm = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(), # Scales image to [0, 1]
])

transform_post_norm = transforms.Compose([
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --- 3. Evaluation Function (Modified for Defense) ---

def evaluate_probe(feature_extractor, probe, loader, device, num_classes, class_names, 
                     perturbation=None, use_fft_defense=False, fft_radius=50,
                     save_samples=False, save_dir=None, num_to_save=0):
    probe.eval()
    
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)
    
    saved_counts = {cls_name: 0 for cls_name in class_names}
    if save_samples:
        print(f"Will save {num_to_save} sample defended images per class to '{save_dir}/'")
        os.makedirs(save_dir, exist_ok=True)
        for cls_name in class_names:
            os.makedirs(os.path.join(save_dir, cls_name), exist_ok=True)
    
    if perturbation is not None:
        perturbation = perturbation.to(device)
        print("Evaluating with ATTACK...")
        if use_fft_defense:
            print(f"Applying FFT Defense with RADIUS = {fft_radius}")
    else:
        print("Evaluating on CLEAN data...")

    with torch.no_grad():
        for images_01, labels, img_paths in tqdm(loader):
            images_01, labels = images_01.to(device), labels.to(device)
            
            defended_images_01 = None
            eval_images_01 = images_01 # Start with clean images
            
            # --- Apply Attack (if provided) ---
            if perturbation is not None:
                eval_images_01 = torch.clamp(images_01 + perturbation, 0.0, 1.0)
            
            # --- Apply Defense (if specified) ---
            if use_fft_defense:
                eval_images_01 = apply_fft_filter_batch(eval_images_01, radius=fft_radius)
                defended_images_01 = eval_images_01 # For saving
            
            # Normalize the (potentially attacked/defended) [0, 1] images
            images_norm = transform_post_norm(eval_images_01)
            
            # --- Get Features and Classify ---
            features = feature_extractor(images_norm)
            outputs = probe(features)
            
            _, predicted = torch.max(outputs.data, 1)
            
            # --- Update Overall and Per-Class Metrics ---
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            for i in range(num_classes):
                class_mask = (labels == i)
                class_total[i] += class_mask.sum().item()
                class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()

            # --- Save Sample Images ---
            if save_samples and (defended_images_01 is not None):
                for i in range(images_01.shape[0]):
                    label_idx = labels[i].item()
                    class_name = class_names[label_idx]
                    
                    if saved_counts[class_name] < num_to_save:
                        img_tensor = defended_images_01[i]
                        original_filename = os.path.basename(img_paths[i])
                        save_path = os.path.join(
                            save_dir, 
                            class_name, 
                            original_filename
                        )
                        utils.save_image(img_tensor, save_path)
                        saved_counts[class_name] += 1
                        
    overall_accuracy = total_correct / total_samples
    per_class_accuracy = class_correct / (class_total + 1e-6) 
    per_class_dict = {class_names[i]: per_class_accuracy[i].item() for i in range(num_classes)}
    
    return overall_accuracy, per_class_dict


# --- 4. Plotting Function ---

def plot_accuracies(per_class_acc_dict, overall_acc, title, save_path):
    if plt is None or sns is None:
        print(f"Skipping plot generation: {title}. Install matplotlib and seaborn.")
        return
    # ... (Plotting code is identical to Script 1) ...
    class_names = list(per_class_acc_dict.keys())
    accuracies = list(per_class_acc_dict.values())
    all_names = class_names + ['Overall']
    all_accs = accuracies + [overall_acc]
    colors = sns.color_palette("viridis", len(all_names))
    colors[-1] = (1.0, 0.0, 0.0)
    plt.figure(figsize=(16, 9))
    sns.barplot(x=all_names, y=all_accs, palette=colors, hue=all_names, legend=False)
    plt.title(title, fontsize=20, pad=20)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.05)
    for i, acc in enumerate(all_accs):
        plt.text(i, acc + 0.01, f'{acc*100:.1f}%', ha='center', fontsize=11, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


# --- 5. Main Execution ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Load Feature Extractor
    feature_extractor = get_model(DEVICE)
    
    # 2. Setup Datasets and Loaders
    print("Setting up validation dataset...")
    val_dataset = HistologyProbeDataset(VAL_DATA_DIR, transform_pre_norm)
    
    num_classes = len(val_dataset.classes)
    class_names = val_dataset.classes
    
    val_loader_eval = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # 3. Initialize and Load Probe
    print(f"Loading pre-trained probe from {PROBE_PATH}...")
    probe = LinearProbe(FEATURE_DIM, num_classes).to(DEVICE)
    try:
        probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Probe file not found at {PROBE_PATH}")
        print("Please run the 'train_and_benchmark.py' script first.")
        exit()
    
    # 4. Benchmark (Clean - for baseline)
    accuracy_clean, per_class_clean = evaluate_probe(
        feature_extractor, probe, val_loader_eval, DEVICE, 
        num_classes=num_classes, class_names=class_names, perturbation=None
    )
    
    # 5. Benchmark (Attacked + Defended)
    try:
        print(f"Loading perturbation from {PERTURBATION_PATH}")
        perturbation = torch.load(PERTURBATION_PATH, map_location=DEVICE)
        
        accuracy_defended, per_class_defended = evaluate_probe(
            feature_extractor, probe, val_loader_eval, DEVICE, 
            num_classes=num_classes, class_names=class_names, perturbation=perturbation,
            use_fft_defense=True, fft_radius=FFT_RADIUS,
            save_samples=True, save_dir=SAMPLES_SAVE_DIR, num_to_save=NUM_SAMPLES_TO_SAVE
        )
        
        plot_accuracies(
            per_class_defended, accuracy_defended, 
            f"Accuracy on DEFENDED Data (FFT Radius={FFT_RADIUS})", "accuracy_defended_fft.png"
        )
        
        # 6. Report Final Results
        print("\n--- 📊 Final FFT Defense Benchmark ---")
        print(f"Accuracy on CLEAN validation data:     {accuracy_clean * 100:.2f}%")
        print(f"Accuracy on DEFENDED validation data: {accuracy_defended * 100:.2f}%")
        print("-------------------------------------")
        print("Per-class accuracy (DEFENDED):")
        for cls, acc in per_class_defended.items():
            print(f"  - {cls}: {acc*100:.2f}%")

    except FileNotFoundError:
        print(f"\nCould not find perturbation file: {PERTURBATION_PATH}")
        print("Skipping defense benchmark.")
        print("-------------------------------------")
    except Exception as e:
        print(f"An error occurred during the defense benchmark: {e}")
