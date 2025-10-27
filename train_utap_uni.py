import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import timm
from huggingface_hub import login, hf_hub_download
from PIL import Image
from tqdm import tqdm

# --- Hyperparameters from the paper ---
# Directory for the training dataset (assuming a structure like the validation one)
TRAIN_DATA_DIR = "/scratch/hmaurya/PGD/NCT-CRC-HE-100K" # <-- IMPORTANT: Change this to your TRAINING data folder
N_CLASSES = 9
N_IMAGES_PER_CLASS = 100
N_TOTAL_IMAGES = 900 # 9 classes * 100 images/class
IMAGE_SIZE = 224
L_EPOCHS = 10
B_BATCH_SIZE = 5

# --- CRITICAL: Scaled Perturbation Parameters ---
# Epsilon (epsilon) is 20 in the [0, 255] pixel space 
EPSILON_RAW = 20.0
# We scale it to the [0, 1] space to match our image tensors
EPSILON_SCALED = EPSILON_RAW / 255.0 
# Step size alpha, as specified in the paper 
ALPHA = 4.35e-4 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Model Loading (from your snippet) ---

def get_model(device):
    """
    Downloads and initializes the UNI2-h model.
    """
    print("Logging into Hugging Face...")
    # login() # Uncomment this and login with your token if needed
    
    print("Downloading UNI2-h model...")
    local_dir = "./assets/ckpts/uni2-h/"
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(
        "MahmoodLab/UNI2-h", 
        filename="pytorch_model.bin", 
        local_dir=local_dir, 
        force_download=True
    )
    
    print("Initializing TIMM model...")
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': IMAGE_SIZE,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model(**timm_kwargs)
    
    print("Loading model state dict...")
    model.load_state_dict(
        torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), 
        strict=True
    )
    
    model.to(device)
    model.eval() # Set to evaluation mode
    
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    print("Model loaded and frozen.")
    return model

# --- 2. Custom Dataset ---

class HistologyDataset(Dataset):
    """
    Dataset to load N_IMAGES_PER_CLASS from each subfolder.
    """
    def __init__(self, data_dir, transform_pre_norm, n_images_per_class):
        self.transform_pre_norm = transform_pre_norm
        self.image_paths = []
        
        # Find class directories (ADI, BACK, DEB, etc.)
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found {len(class_dirs)} classes: {class_dirs}")

        if len(class_dirs) != N_CLASSES:
            print(f"Warning: Expected {N_CLASSES} classes, but found {len(class_dirs)}.")

        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            # Get all images in the class folder (adjust extension if needed)
            all_images = glob.glob(os.path.join(class_path, "*.tif"))
            all_images.extend(glob.glob(os.path.join(class_path, "*.jpg")))
            
            # Sample N images from this class
            if len(all_images) >= n_images_per_class:
                sampled_images = random.sample(all_images, n_images_per_class)
            else:
                print(f"Warning: Class {class_dir} has only {len(all_images)} images. Using all of them.")
                sampled_images = all_images
                
            self.image_paths.extend(sampled_images)
            
        random.shuffle(self.image_paths)
        print(f"Created dataset with {len(self.image_paths)} total images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # Apply Resize, Crop, and ToTensor (scales to [0, 1])
        image_tensor = self.transform_pre_norm(image)
        return image_tensor

# --- 3. Main Training Function ---

def train_utap():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # --- Setup Model ---
    model = get_model(device)

    # --- Setup Data ---
    # Transform to get [0, 1] tensor
    transform_pre_norm = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), # Scales image from [0, 255] to [0, 1]
    ])
    
    # Transform to normalize [0, 1] tensor for the model
    transform_post_norm = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dataset = HistologyDataset(
        data_dir=TRAIN_DATA_DIR,
        transform_pre_norm=transform_pre_norm,
        n_images_per_class=N_IMAGES_PER_CLASS
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=B_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # --- Initialize Perturbation ---
    # **FIX:** Delta now lives in the [0, 1] scaled space.
    # Its range will be clamped to [-EPSILON_SCALED, EPSILON_SCALED]
    delta = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, requires_grad=True)

    # --- Loss Function ---
    # We want to MINIMIZE the cosine similarity [cite: 959, 1301]
    cosine_sim_loss = nn.CosineSimilarity(dim=1)

    print("\n--- Starting UTAP Perturbation Training ---")
    print(f"Epochs: {L_EPOCHS}, Batch Size: {B_BATCH_SIZE}")
    print(f"Epsilon (raw): {EPSILON_RAW}, Epsilon (scaled): {EPSILON_SCALED:.6f}")
    print(f"Alpha (step size): {ALPHA:.2e}")
    
    for epoch in range(L_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{L_EPOCHS}")
        total_loss = 0.0
        
        pbar = tqdm(dataloader)
        for images_01 in pbar:
            images_01 = images_01.to(device)
            B, C, H, W = images_01.shape

            # --- Create Attacked Images ---
            # **FIX:** Add delta (which is already scaled) directly to the [0, 1] image
            # We broadcast the (1, C, H, W) delta to (B, C, H, W)
            attacked_images_01 = torch.clamp(images_01 + delta, 0.0, 1.0)
            
            # 2. Normalize both original and attacked images for the model
            original_norm = transform_post_norm(images_01)
            attacked_norm = transform_post_norm(attacked_images_01)
            
            # --- Forward Pass ---
            # Get original features without building a computation graph
            with torch.no_grad():
                original_features = model(original_norm)
            
            # Get attacked features WITH the computation graph
            attacked_features = model(attacked_norm)
            
            # --- Calculate Loss ---
            # Minimize the mean cosine similarity [cite: 959, 1301]
            loss = cosine_sim_loss(original_features, attacked_features).mean()
            
            # --- Backward Pass ---
            if delta.grad is not None:
                delta.grad.zero_()
            
            loss.backward()
            
            # --- PGD Update ---
            # Gradient DESCENT to MINIMIZE the loss
            with torch.no_grad():
                grad = delta.grad.data
                
                # Update delta using the sign of the gradient
                delta.data = delta.data - ALPHA * torch.sign(grad)
                
                # **FIX:** Clamp/Project delta to stay within the SCALED budget
                delta.data = torch.clamp(delta.data, -EPSILON_SCALED, EPSILON_SCALED)

            total_loss += loss.item()
            pbar.set_description(f"Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Cosine Similarity (Loss): {avg_loss:.6f}")

    print("\n--- Training Complete ---")
    
    # --- Save the Perturbation ---
    output_path_pt = "utap_perturbation.pt"
    output_path_png = "utap_perturbation_visualization.png"
    
    # Save the 'delta' tensor (which is in the [-20/255, 20/255] range)
    torch.save(delta.detach().cpu(), output_path_pt)
    print(f"Saved perturbation tensor to: {output_path_pt}")
    
    # **FIX:** Visualize by normalizing from [-EPSILON_SCALED, EPSILON_SCALED] to [0, 1]
    vis_delta = (delta.detach().cpu() + EPSILON_SCALED) / (2 * EPSILON_SCALED)
    save_image(vis_delta, output_path_png)
    print(f"Saved perturbation visualization to: {output_path_png}")


if __name__ == "__main__":
    train_utap()
