import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# --- CONCH Specific Imports ---
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
# --------------------------
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
PERTURBATION_PATH = "conch_utap_textalign_perturbation.pt"

# Save a few attacked samples (like your UNI2 example)
SAVE_SAMPLES = True
SAMPLES_SAVE_DIR = "conch_attacked_samples_orig_name_text_align"  # per-class subfolders, original filenames
NUM_SAMPLES_TO_SAVE = 5  # number of attacked images to save per class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224 # CONCH default is 224, make sure perturbation matches if different
BATCH_SIZE = 64 # Adjust based on your VRAM

# --- Mapping from Abbreviations to Full Names ---
CLASS_NAME_MAPPING = {
    'ADI': 'Adipose',
    'BACK': 'Background',
    'DEB': 'Debris',
    'LYM': 'Lymphocytes',
    'MUC': 'Mucus',
    'MUS': 'Smooth Muscle',
    'NORM': 'Normal',
    'STR': 'Stroma',
    'TUM': 'Tumor'
}
# ---------------------------------------------

# --- 1. Model & Data Setup ---

def get_conch_model_and_transform(device):
    """ Loads the CONCH model and its specific evaluation transform. """
    print("Loading CONCH model...")
    # Using 'conch_ViT-B-16' as specified
    # checkpoint_path="hf_hub:MahmoodLab/conch" loads the default weights
    model, eval_transform = create_model_from_pretrained(
        'conch_ViT-B-16',
        checkpoint_path="hf_hub:MahmoodLab/conch"
    )
    model = model.to(device).eval()
    print("CONCH model loaded.")
    return model, eval_transform

class HistologyProbeDataset(Dataset):
    # --- MODIFIED: Uses the transform passed during init ---
    def __init__(self, data_dir, transform):
        self.transform = transform # Use the transform provided (CONCH's transform)
        self.samples = []
        class_dirs = sorted([d for d in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, d))])

        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}
        self.classes = class_dirs # KEEP ABBREVIATIONS HERE FOR FOLDER MATCHING
        print(f"Found {len(self.classes)} classes in {data_dir}: {self.class_to_idx}")

        for class_name, label_idx in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            # Look for tif and jpg (add png if you need)
            for img_ext in ["*.tif", "*.jpg"]:
                for img_file in glob.glob(os.path.join(class_path, img_ext)):
                    self.samples.append((img_file, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # --- MODIFICATION: Apply transform directly ---
            image = Image.open(img_path).convert("RGB")
            # CONCH's transform handles resize, crop, ToTensor, AND Normalize
            image_tensor = self.transform(image)
            return image_tensor, label, img_path
        except Exception as e:
            print(f"Warning: Error loading {img_path}, skipping. Error: {e}")
            # Need a placeholder that matches the expected output shape and type
            placeholder_tensor = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            return placeholder_tensor, 0, "error_image.png"

# --- 2. Evaluation Function for CONCH ---

def evaluate_conch(model, text_embeddings, loader, device, num_classes, class_names,
                   perturbation=None, save_samples=False, save_dir=None, num_to_save=0):
    """
    If perturbation is provided and save_samples=True, saves up to `num_to_save`
    attacked images per class into per-class subfolders under `save_dir`,
    preserving the original filenames.
    """
    model.eval() # Ensure model is in eval mode

    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)

    # Prepare directories and per-class counters (use abbreviations for folder names)
    saved_counts = None
    if save_samples and perturbation is not None:
        os.makedirs(save_dir, exist_ok=True)
        saved_counts = {cls: 0 for cls in class_names}
        for cls in class_names:
            os.makedirs(os.path.join(save_dir, cls), exist_ok=True)
        print(f"Will save up to {num_to_save} attacked samples per class into '{save_dir}/<CLASS>/'")

    if perturbation is not None:
        perturbation = perturbation.to(device)
        print("Evaluating with ATTACK...")
        # --- Need transform to convert PIL image to [0,1] tensor BEFORE adding perturbation ---
        to_tensor_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor() # Scales image to [0, 1]
        ])
        # Get the CONCH normalization transform separately
        conch_normalize = loader.dataset.transform.transforms[-1] # Assumes Normalize is last
        # -------------------------------------------------------------------------------------
    else:
        print("Evaluating on CLEAN data...")

    with torch.no_grad():
        for image_tensors_normalized, labels, img_paths in tqdm(loader):

            eval_tensors_normalized = image_tensors_normalized.to(device)
            labels = labels.to(device)

            # --- Apply Attack (if provided) ---
            valid_indices = None
            attacked_images_01 = None
            if perturbation is not None:
                # Need to load original images, convert to [0,1], add pert, then normalize
                batch_images_01 = []
                valid_indices = [] # Keep track of which images loaded successfully
                for idx, img_path in enumerate(img_paths):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_01 = to_tensor_transform(img)
                        batch_images_01.append(img_01)
                        valid_indices.append(idx)
                    except Exception as e: # Handle loading errors during attack phase too
                        print(f"Warning: Error loading {img_path} during attack eval, skipping sample in batch. Error: {e}")
                        # Skip this image by not adding its index to valid_indices

                if not batch_images_01: # Skip batch if all images failed
                    continue

                images_01_batch = torch.stack(batch_images_01).to(device)
                attacked_images_01 = torch.clamp(images_01_batch + perturbation, 0.0, 1.0)
                # Now normalize the attacked [0,1] images for CONCH
                eval_tensors_normalized = conch_normalize(attacked_images_01)
                # Filter labels to only include those for successfully loaded images
                labels = labels[valid_indices]
                # Also filter img_paths to correspond to valid_indices for saving
                img_paths = [img_paths[i] for i in valid_indices]
                if labels.numel() == 0: # Skip if no valid labels left
                    continue

            # --- Get Image Embeddings ---
            # Input should be the already transformed+normalized tensor from dataloader or attacked+normalized tensor
            image_embeddings = model.encode_image(eval_tensors_normalized) # Add normalize=True if needed by model version

            # --- Calculate Similarities ---
            # image_embeddings shape: [Batch_valid, EmbedDim]
            # text_embeddings shape: [NumClasses, EmbedDim]
            similarities = image_embeddings @ text_embeddings.T # Result shape: [Batch_valid, NumClasses]

            # --- Get Predictions ---
            predicted_indices = torch.argmax(similarities, dim=1)

            # --- Update Overall and Per-Class Metrics ---
            current_batch_size = labels.size(0) # Use the size after filtering
            total_samples += current_batch_size
            total_correct += (predicted_indices == labels).sum().item()

            for i in range(num_classes):
                class_mask = (labels == i)
                class_total[i] += class_mask.sum().item()
                class_correct[i] += (predicted_indices[class_mask] == labels[class_mask]).sum().item()

            # --- Save Sample Images (only if attacked) ---
            if (perturbation is not None) and save_samples and (attacked_images_01 is not None):
                for i in range(attacked_images_01.size(0)):
                    label_idx = labels[i].item()
                    cls_name = class_names[label_idx]  # use abbrev for folder, like ADI, TUM, etc.
                    if saved_counts[cls_name] < num_to_save:
                        img_tensor = attacked_images_01[i].detach().cpu()
                        original_filename = os.path.basename(img_paths[i])
                        save_path = os.path.join(save_dir, cls_name, original_filename)
                        utils.save_image(img_tensor, save_path)
                        saved_counts[cls_name] += 1

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    per_class_accuracy = class_correct / (class_total + 1e-6)
    # --- MODIFICATION: Use MAPPING for display names (pretty print) ---
    per_class_dict = {CLASS_NAME_MAPPING.get(class_names[i], class_names[i]): per_class_accuracy[i].item()
                      for i in range(num_classes)}
    # ---------------------------------------------------

    return overall_accuracy, per_class_dict


# --- 3. Plotting Function ---

def plot_accuracies(per_class_acc_dict, overall_acc, title, save_path):
    if plt is None or sns is None:
        print(f"Skipping plot generation: {title}. Install matplotlib and seaborn.")
        return
    # --- MODIFICATION: Keys are now full names ---
    class_names_full = list(per_class_acc_dict.keys())
    accuracies = list(per_class_acc_dict.values())
    # ---------------------------------------------
    all_names = class_names_full + ['Overall'] # Use full names for plot labels
    all_accs = accuracies + [overall_acc]

    colors = sns.color_palette("viridis", len(all_names))
    colors[-1] = (1.0, 0.0, 0.0)
    plt.figure(figsize=(18, 10)) # Increased figure size for longer labels
    sns.barplot(x=all_names, y=all_accs, palette=colors, hue=all_names, legend=False)
    plt.title(title, fontsize=20, pad=20)
    plt.ylabel('Accuracy (based on Max CosSim)', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11) # Adjust font size if needed
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.05)
    for i, acc in enumerate(all_accs):
        plt.text(i, acc + 0.01, f'{acc*100:.1f}%', ha='center', fontsize=10, weight='bold') # Adjust font size
    plt.tight_layout(pad=1.5) # Add padding
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


# --- 4. Main Execution ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load CONCH Model and Transform
    model, conch_eval_transform = get_conch_model_and_transform(DEVICE)
    tokenizer = get_tokenizer()

    # 2. Setup Validation Dataset and Loader (using CONCH transform)
    print("Setting up validation dataset...")
    val_dataset = HistologyProbeDataset(VAL_DATA_DIR, conch_eval_transform)

    num_classes = len(val_dataset.classes)
    class_names_abbrev = val_dataset.classes # Keep abbreviations for internal logic

    val_loader_eval = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 3. Generate Text Prompts and Embeddings (using FULL names)
    print("Generating text prompts with full names...")
    # --- MODIFICATION: Use MAPPING to get full names ---
    prompts = []
    ordered_full_names = [] # Store full names in the order of class_indices
    for i in range(num_classes):
        abbrev_name = class_names_abbrev[i]
        full_name = CLASS_NAME_MAPPING.get(abbrev_name, abbrev_name) # Fallback to abbrev if not found
        prompts.append(f"This is an image of {full_name}")
        ordered_full_names.append(full_name) # Keep track of full names for plotting keys
    # ----------------------------------------------------

    print("Prompts being used:")
    for p in prompts: print(f"- {p}")

    tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(DEVICE)

    print("Computing text embeddings...")
    with torch.no_grad():
        text_embeddings = model.encode_text(tokenized_prompts) # Add normalize=True if needed

    # 4. Benchmark (Before Attack)
    accuracy_before, per_class_before = evaluate_conch(
        model, text_embeddings, val_loader_eval, DEVICE,
        num_classes=num_classes, class_names=class_names_abbrev, # Pass abbrevs for internal logic
        perturbation=None
    )

    # Use full names for plotting keys
    plot_accuracies(
        per_class_before, accuracy_before,
        "CONCH Accuracy (Max CosSim) on CLEAN Validation Data",
        "conch_utap_text_align_accuracy_conch_clean_fullname.png"
    )

    # 5. Benchmark (After Attack)
    try:
        print(f"Loading perturbation from {PERTURBATION_PATH}")
        # Ensure perturbation is loaded to the correct device
        perturbation = torch.load(PERTURBATION_PATH, map_location='cpu').to(DEVICE)

        accuracy_after, per_class_after = evaluate_conch(
            model, text_embeddings, val_loader_eval, DEVICE,
            num_classes=num_classes, class_names=class_names_abbrev, # Pass abbrevs for internal logic
            perturbation=perturbation,
            save_samples=SAVE_SAMPLES, save_dir=SAMPLES_SAVE_DIR, num_to_save=NUM_SAMPLES_TO_SAVE
        )

        # Use full names for plotting keys
        plot_accuracies(
            per_class_after, accuracy_after,
            "CONCH Accuracy (Max CosSim) on ATTACKED Validation Data",
            "conch_utap_text_align_accuracy_conch_attacked_fullname.png"
        )

        # 6. Report Final Results
        print("\n--- 📊 Final CONCH Benchmark Results ---")
        print(f"Accuracy (Max CosSim) on CLEAN validation data:   {accuracy_before * 100:.2f}%")
        print(f"Accuracy (Max CosSim) on ATTACKED validation data: {accuracy_after * 100:.2f}%")
        print("-------------------------------------")
        print("Per-class accuracy (CLEAN):")
        # Use full names for printing
        for full_name in ordered_full_names:
            acc = per_class_before.get(full_name, 0.0) # Get acc using full name
            print(f"  - {full_name}: {acc*100:.2f}%")
        print("\nPer-class accuracy (ATTACKED):")
        # Use full names for printing
        for full_name in ordered_full_names:
            acc = per_class_after.get(full_name, 0.0) # Get acc using full name
            print(f"  - {full_name}: {acc*100:.2f}%")

        if SAVE_SAMPLES:
            print(f"\nAttacked samples saved under: {SAMPLES_SAVE_DIR}/<CLASS>/ (up to {NUM_SAMPLES_TO_SAVE} each)")

    except FileNotFoundError:
        print(f"\n--- 📊 Final CONCH Benchmark Results ---")
        print(f"Accuracy (Max CosSim) on CLEAN validation data: {accuracy_before * 100:.2f}%")
        print("Per-class accuracy (CLEAN):")
        for full_name in ordered_full_names:
            acc = per_class_before.get(full_name, 0.0)
            print(f"  - {full_name}: {acc*100:.2f}%")
        print(f"\nCould not find perturbation file: {PERTURBATION_PATH}")
        print("Skipping attack benchmark.")
        print("-------------------------------------")
    except Exception as e:
        print(f"An error occurred during the attack benchmark: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
