import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from tail_mil import TAIL_MIL 
from dataset import WormTrajectoryDataset

# --- Configuration ---
DATA_PATH = "../preprocessed_data/" 
SEGMENT_LEN = 900

MAX_SEGMENTS = 150
EPOCHS = 200
PATIENCE = 10      # Early stopping patience
K_FOLDS = 5        # Number of folds for Cross-Validation

def train(batch_size, learning_rate, embed_dim):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    print(f"Configuration: Batch Size={batch_size}, LR={learning_rate}, Embed Dim={embed_dim}, Epochs={EPOCHS}")
    # 2. Prepare Data
    full_dataset = WormTrajectoryDataset(root_dir=DATA_PATH, max_segments=MAX_SEGMENTS, segment_len=SEGMENT_LEN)
    labels = np.array(full_dataset.labels) # Get labels for stratification

    # Initialize K-Fold
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Store results across folds
    fold_results = {
        'best_val_acc': [],
        'best_val_f1': [],
        'best_val_auc': [],
        'train_loss_history': [],
        'val_loss_history': []
    }

    print(f"Starting {K_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
        
        # Create Subsets and Loaders
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 3. Initialize Model (Fresh for each fold)
        model = TAIL_MIL(segment_len=SEGMENT_LEN, num_vars=3, embed_dim=embed_dim).to(device)
        
        # 4. Optimizer & Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss() 

        # Early stopping state for this fold
        best_val_acc = -float("inf")
        best_val_f1 = 0
        best_val_auc = 0
        epochs_no_improve = 0
        best_epoch = -1
        
        # History for this fold
        fold_train_losses = []
        fold_val_losses = []

        # --- Loop ---
        for epoch in tqdm(range(EPOCHS), desc=f"Fold {fold+1} Epochs"):
            model.train()
            train_loss = 0
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                
                preds, _, _ = model(X)
                preds = preds.squeeze()
                
                loss = criterion(preds, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            fold_train_losses.append(avg_train_loss)
            
            # --- Validation ---
            model.eval()
            val_loss = 0
            val_labels = []
            val_preds = []
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    preds, _, _ = model(X)
                    preds = preds.squeeze()
                    
                    loss = criterion(preds, y)
                    val_loss += loss.item()
                    
                    val_labels.extend(y.cpu().detach().numpy())
                    val_preds.extend(preds.cpu().detach().numpy())

            avg_val_loss = val_loss / len(val_loader)
            fold_val_losses.append(avg_val_loss)

            # Metrics
            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            val_acc = accuracy_score(val_labels, val_preds_binary)
            val_f1 = f1_score(val_labels, val_preds_binary)
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except:
                val_auc = 0.5

            # Pretty print
            tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping logic (monitoring val accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                epochs_no_improve = 0
                best_epoch = epoch + 1
                # Save best model for this fold
                torch.save(model.state_dict(), f"tail_mil_worm_fold{fold+1}_best.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered for Fold {fold+1}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}.")
                break
        
        # Store fold results
        fold_results['best_val_acc'].append(best_val_acc)
        fold_results['best_val_f1'].append(best_val_f1)
        fold_results['best_val_auc'].append(best_val_auc)
        fold_results['train_loss_history'].append(fold_train_losses)
        fold_results['val_loss_history'].append(fold_val_losses)

    # --- Summary ---
    print("\n=== Cross-Validation Summary ===")
    print(f"Average Val Acc: {np.mean(fold_results['best_val_acc']):.4f} (+/- {np.std(fold_results['best_val_acc']):.4f})")
    print(f"Average Val F1: {np.mean(fold_results['best_val_f1']):.4f} (+/- {np.std(fold_results['best_val_f1']):.4f})")
    print(f"Average Val AUC: {np.mean(fold_results['best_val_auc']):.4f} (+/- {np.std(fold_results['best_val_auc']):.4f})")

    # Plotting Average Loss
    plt.figure(figsize=(10, 6))
    
    # Plot each fold
    for i in range(K_FOLDS):
        plt.plot(fold_results['train_loss_history'][i], alpha=0.3, color='blue', linestyle='--')
        plt.plot(fold_results['val_loss_history'][i], alpha=0.3, color='orange', linestyle='--')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='orange', lw=2)]
    
    plt.title(f'{K_FOLDS}-Fold CV Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(custom_lines, ['Train Loss (All Folds)', 'Val Loss (All Folds)'])
    plt.grid(True)
    plt.savefig('cv_training_validation_loss.png')
    print("CV Loss plot saved to 'cv_training_validation_loss.png'")
    return fold_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TAIL_MIL model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--grid_search", type=str, default=None, help="grid search parameters, in the form [[batch_size1, batch_size2], [learning_rate1, learning_rate2], ...]")
    
    args = parser.parse_args()
    
    if args.grid_search is not None:
        import itertools
        import ast
        param_names = ['batch_size', 'learning_rate', 'embed_dim', 'epochs']
        grid_search_params = ast.literal_eval(args.grid_search)
        param_combinations = list(itertools.product(*grid_search_params))
        
        all_results = []
        for param_set in param_combinations:
            params = dict(zip(param_names, param_set))
            print(f"\n========= Grid Search: Training with parameters: {params} =============")
            result = train(batch_size=params['batch_size'], learning_rate=params['learning_rate'], embed_dim=params['embed_dim'])
            all_results.append((params, result))
        
        # summarize grid search results
        print("\n=== Grid Search Summary ===")
        for params, result in all_results:
            print(f"Params: {params} => Avg Val Acc: {np.mean(result['best_val_acc']):.4f}, Avg Val F1: {np.mean(result['best_val_f1']):.4f}, Avg Val AUC: {np.mean(result['best_val_auc']):.4f}")
    else:
        train(batch_size=args.batch_size, learning_rate=args.learning_rate, embed_dim=args.embed_dim, epochs=args.epochs)