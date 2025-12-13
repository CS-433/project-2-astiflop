import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# ==========================================
# 1. Dataset Class
# ==========================================

class CElegansCNNDataset(Dataset):
    """
    Dataset for C. Elegans CNN.
    Loads images from the directory structure:
    root_dir/
      Treatment_Group/
        Worm_ID/
           photos_trajectories/
             *.png
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # list of (image_path, label, worm_id)
        
        self.class_map = {
            "TERBINAFINE- (control)": 0, 
            "TERBINAFINE+": 1
        }
        
        self._load_samples()
        
    def _load_samples(self):
        # Traverse categories
        for treatment_name, label in self.class_map.items():
            treatment_path = os.path.join(self.root_dir, treatment_name)
            if not os.path.exists(treatment_path):
                print(f"Warning: Path not found: {treatment_path}")
                continue
                
            # Traverse worms
            worm_dirs = [d for d in os.listdir(treatment_path) if os.path.isdir(os.path.join(treatment_path, d))]
            
            for worm_id in worm_dirs:
                img_dir = os.path.join(treatment_path, worm_id, "photos_trajectories")
                if not os.path.exists(img_dir):
                    continue
                    
                # Get all png images
                images = glob.glob(os.path.join(img_dir, "*.png"))
                for img_path in images:
                    self.samples.append((img_path, label, worm_id))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, worm_id = self.samples[idx]
        
        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
             # Handle missing/corrupt images gracefully?
             # For now raise error
             raise ValueError(f"Failed to load image: {img_path}")
             
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for Torchvision transforms if not already handled by ToTensor logic within transform
        # Most torchvision transforms expect PIL Image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32), worm_id

    def get_indices_labels_groups(self):
        """Helper to get all labels and groups for splitting"""
        labels = [s[1] for s in self.samples]
        groups = [s[2] for s in self.samples] # worm_ids
        return np.arange(len(self.samples)), np.array(labels), np.array(groups)

# ==========================================
# 2. ResNet Model
# ==========================================

def get_resnet_model(pretrained=True, freeze_layers=False):
    """
    Returns a ResNet18 model adapted for binary classification.
    """
    # Load ResNet18
    # 'pretrained=True' is deprecated in newer versions, using 'weights' is better, 
    # but for Colab compatibility (which might have older/newer torch), we stick to new API if possible or fallback
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=pretrained)
    
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
            
    # Modify the final layer (fc)
    # ResNet18 fc input features is 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    return model

# ==========================================
# 3. Training Loop
# ==========================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_worm_ids = []
    
    with torch.no_grad():
        for images, labels, worm_ids in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
            all_worm_ids.extend(worm_ids)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # --- Image Level Metrics ---
    all_preds_bin_img = (all_preds > 0.5).astype(int)
    acc_img = accuracy_score(all_labels, all_preds_bin_img)
    f1_img = f1_score(all_labels, all_preds_bin_img, zero_division=0)
    
    # --- Worm Level Metrics (Majority Voting / Soft Voting) ---
    # Group by worm_id
    worm_preds = {}
    worm_labels = {}
    
    for i, wid in enumerate(all_worm_ids):
        if wid not in worm_preds:
            worm_preds[wid] = []
            worm_labels[wid] = all_labels[i] # Label is constant per worm
        worm_preds[wid].append(all_preds[i])
        
    # Aggregate
    final_worm_labels = []
    final_worm_preds = []
    
    for wid in worm_preds:
        # Soft voting: average probability
        avg_prob = np.mean(worm_preds[wid])
        final_worm_preds.append(avg_prob)
        final_worm_labels.append(worm_labels[wid])
        
    final_worm_labels = np.array(final_worm_labels)
    final_worm_preds = np.array(final_worm_preds)
    final_worm_preds_bin = (final_worm_preds > 0.5).astype(int)
    
    acc_worm = accuracy_score(final_worm_labels, final_worm_preds_bin)
    f1_worm = f1_score(final_worm_labels, final_worm_preds_bin, zero_division=0)
    
    return acc_worm, f1_worm, acc_img, f1_img

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    import argparse
    
    # --- HYPERPARAMETERS ---
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LR = 1e-4
    DEFAULT_EPOCHS = 15
    IMG_SIZE = 128
    
    parser = argparse.ArgumentParser(description="Train ResNet on C. Elegans Dataset with Statified Group K-Fold")
    parser.add_argument("--data_dir", type=str, default="cnn_dataset", help="Path to cnn_dataset folder")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for Cross Validation")
    
    # Try to parse known args to ignore issues in Colab/Jupyter environs if run as script
    args, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- TRANSFORMS (Data Augmentation) ---
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180), # Orientation invarianc
        transforms.ToTensor(), # scale [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])
    
    # Validation transforms (No augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- DATASET ---
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
    else:
        # Load dataset WITHOUT transforms first, we apply them in the loop/subset logic?
        # Actually Dataset class applies transform. We need separate datasets for Train/Val 
        # because of different transforms.
        # So we will create two dataset instances pointing to same data, but different transforms.
        
        dataset_full = CElegansCNNDataset(args.data_dir, transform=None)
        
        if len(dataset_full) == 0:
            print("Error: No images found.")
            exit()
            
        print(f"Total samples: {len(dataset_full)}")
        
        # Get indices and groups for splitting
        indices, labels, groups = dataset_full.get_indices_labels_groups()
        
        # Cross Validation
        sgkf = StratifiedGroupKFold(n_splits=args.n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(indices, labels, groups=groups)):
            print(f"\n{'='*20} FOLD {fold+1}/{args.n_splits} {'='*20}")
            
            # Create Subsets with appropriate transforms
            # We use the custom class wrapper approach or just create two dataset objs
            # Re-instantiating dataset is cheap (just file listing), but let's be efficient.
            # We can just set the transform on the dataset if we are careful, 
            # OR cleaner: Subset wrapper that applies transform.
            
            class TransformedSubset(Dataset):
                def __init__(self, subset, transform=None):
                    self.subset = subset
                    self.transform = transform
                def __getitem__(self, index):
                    img, label, worm_id = self.subset[index]
                    # dataset returns PIL because transform=None in dataset_full
                    if self.transform:
                        img = self.transform(img)
                    return img, label, worm_id
                def __len__(self):
                    return len(self.subset)
            
            train_subset = torch.utils.data.Subset(dataset_full, train_idx)
            val_subset = torch.utils.data.Subset(dataset_full, val_idx)
            
            train_ds = TransformedSubset(train_subset, transform=train_transform)
            val_ds = TransformedSubset(val_subset, transform=val_transform)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            
            # Initialize Model
            model = get_resnet_model(pretrained=True).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr) # AdamW usually better
            
            best_fold_f1 = 0.0
            
            for epoch in range(args.epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_acc, val_f1, val_acc_img, val_f1_img = validate(model, val_loader, device)
                
                print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc (Worm)={val_acc:.4f}, Val F1 (Worm)={val_f1:.4f} [Img F1: {val_f1_img:.4f}]")
                
                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    # Save best model for this fold
                    torch.save(model.state_dict(), f"resnet_fold_{fold+1}.pth")
            
            print(f"Fold {fold+1} Best F1: {best_fold_f1:.4f}")
            fold_results.append(best_fold_f1)
            
        print(f"\nMean F1 over {args.n_splits} folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
