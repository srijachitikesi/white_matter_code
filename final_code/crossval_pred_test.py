#!/usr/bin/env python
# coding: utf-8


import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from custom_dataset import CustDataset
from network_cbcl import Network
from torch.utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import ShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, explained_variance_score, r2_score


def calculate_metrics(actual, predicted):
    r2_vals = r2_score(actual, predicted)
    ev_vals = explained_variance_score(actual, predicted)
    correlation = np.corrcoef(actual,predicted)[0,1]

    return {'r2 score':r2_vals, 'ev': ev_vals, 'correlation': correlation}


#setup initialization
torch.manual_seed(52)
num_workers = 4
batch_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} device")
# print("batch size", batch_size)
os.system('rm -rf checkpoint_fold_*.pth')
#data initialization
data = CustDataset(transform =
                        transforms.Compose([
                            transforms.RandomHorizontalFlip()
                            ]))

# Initialize lists to store metrics for all folds
mse_list, r2_list, correlation_list, ev_list = [], [], [], []
best_train_mse_list, best_train_r2_list, best_train_correlation_list, best_train_ev_list = [], [], [], []
best_valid_mse_list, best_valid_r2_list, best_valid_correlation_list, best_valid_ev_list = [], [], [], []



# prepare for k-fold
kf = KFold(n_splits=3, shuffle=True, random_state=52)


#Early Stopping
class EarlyStopping:
        def __init__(self, patience=20, verbose=False, delta=0, fold=None, job_id=None):
      
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')
            self.delta = delta
            self.fold = fold
            self.job_id = job_id  # Unique job ID for saving checkpoint

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model for fold {self.fold} and job {self.job_id}...')
                torch.save(model.state_dict(), f'checkpoint_{self.job_id}_fold_{self.fold}.pth')  # Save model per fold
                self.val_loss_min = val_loss

scaler = GradScaler()

learning_rates = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]
mask = 'CF'
fc2 = 256
hidden_dropout = 0.3 # Example of dropout rate
job_id = f"{mask}_{fc2}_{hidden_dropout}_{learning_rates[0]}_{learning_rates[-1]}"

best_valid_loss = float('inf')

for fold, (train_idx, valid_idx) in enumerate(kf.split(data.train_idx)):
   
    print(f"Training fold {fold}...")

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(data.test_idx)

    train_loader = DataLoader(data,batch_size=batch_size,
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(data,batch_size=batch_size,
                                sampler= valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(data,batch_size=batch_size,
                                sampler= test_sampler, num_workers=num_workers)
    model = Network()
    model.to(device)
    # print(model)
    early_stopping = EarlyStopping(patience=20, verbose=True, fold=fold, job_id=job_id)

    

    epochs = 100
    criterion = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    optimizer = torch.optim.Adam(
        [
            {"params": model.cv1.parameters(), "lr": learning_rates[0]},
            {"params": model.bn1.parameters(), "lr": learning_rates[0]},
            {"params": model.cv2.parameters(), "lr": learning_rates[1]},
            {"params": model.bn2.parameters(), "lr": learning_rates[1]},
            {"params": model.cv3.parameters(), "lr": learning_rates[2]},
            {"params": model.bn3.parameters(), "lr": learning_rates[2]},
            {"params": model.cv4.parameters(), "lr": learning_rates[3]},
            {"params": model.bn4.parameters(), "lr": learning_rates[3]},
            # {"params": model.cv5.parameters(), "lr": learning_rates[4]},
            # {"params": model.bn5.parameters(), "lr": learning_rates[4]},
            {"params": model.fc1.parameters(), "lr": learning_rates[5]},
            {"params": model.fc2.parameters(), "lr": learning_rates[5]},
        ],
        lr=5e-4,
    )
    # learning_rate_layer = [1e-4,5e-4,5e-4,1e-5,5e-5]
    # optimizer = torch.optim.Adam(
    #     [{"params": model.cv1.parameters(), "lr": learning_rate_layer[0]},
    #         {"params": model.bn1.parameters(), "lr": learning_rate_layer[0]},
    #         {"params": model.cv2.parameters(), "lr": learning_rate_layer[1]},
    #         {"params": model.bn2.parameters(), "lr": learning_rate_layer[1]},
    #         {"params": model.cv3.parameters(), "lr": learning_rate_layer[2]},
    #         {"params": model.bn3.parameters(), "lr": learning_rate_layer[2]},
    #         {"params": model.cv4.parameters(), "lr": learning_rate_layer[3]},
    #         {"params": model.bn4.parameters(), "lr": learning_rate_layer[3]},
    #         {"params": model.cv5.parameters(), "lr": learning_rate_layer[4]},
    #         {"params": model.bn5.parameters(), "lr": learning_rate_layer[4]},
    #         {"params": model.fc1.parameters(), "lr": learning_rate_layer[4]} ],
    #     lr=5e-5,weight_decay = 1e-4
    # )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)


    #training
    print('Starting to Train...')
    best_loss = float('inf')

    for e in range(1,epochs+1):
        model.train()
        train_loss = 0
        actual_batch_values_train = []
        predicted_batch_values_train  = []

        # with torch.no_grad():
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device).float(), y.to(device).float()
            X = torch.unsqueeze(X, 1).float()
            with autocast():

                pred = model(X)
                pred = pred.squeeze()
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += (loss.item()*X.shape[0])

            if pred.dim() > 0:
                actual_batch_values_train.extend(y.cpu().numpy())
                predicted_batch_values_train.extend(pred.cpu().detach().numpy())
        
        avg_train_loss = train_loss / len(train_idx)
        train_metrics = calculate_metrics(actual_batch_values_train, predicted_batch_values_train)
        
        print(f"Epoch {e}/{epochs} - Average train loss: {avg_train_loss} - R2: {train_metrics['r2 score']:.4f}, EV: {train_metrics['ev']:.4f}, Correlation: {train_metrics['correlation']:.4f}")
        
                   
        #validation
        
        model.eval()
        valid_loss = 0
        actual_batch_values_valid = []
        predicted_batch_values_valid = []

        with torch.no_grad():
            with autocast():

                for X,y in valid_loader:
                    X, y = X.to(device).float(), y.to(device).float()
                    X = torch.unsqueeze(X, 1).float()

                    pred = model(X)
                    pred = pred.squeeze()
                    loss = criterion(pred, y)
                    valid_loss += ((loss.item())*X.shape[0])

                    if pred.dim() > 0:
                        actual_batch_values_valid.extend(y.cpu().numpy())
                        predicted_batch_values_valid.extend(pred.cpu().detach().numpy())

        avg_valid_loss = valid_loss / len(valid_idx)
    
        validation_metrics = calculate_metrics(actual_batch_values_valid, predicted_batch_values_valid)
        print(f"Epoch {e}/{epochs} - Average validation loss: {avg_valid_loss} - R2: {validation_metrics['r2 score']:.4f}, EV: {validation_metrics['ev']:.4f}, Correlation: {validation_metrics['correlation']:.4f}")

       
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {e}")
            break
        
        scheduler.step(avg_valid_loss)  # Monitor validation loss


        if avg_valid_loss < best_valid_loss:
            best_train_loss = avg_train_loss
            best_train_r2 = train_metrics['r2 score']
            best_train_correlation = train_metrics['correlation']
            best_train_ev = train_metrics['ev']

            best_valid_loss = avg_valid_loss
            best_valid_r2 = validation_metrics['r2 score']
            best_valid_correlation = validation_metrics['correlation']
            best_valid_ev = validation_metrics['ev']


    best_train_mse_list.append(best_train_loss)
    best_train_r2_list.append(best_train_r2)
    best_train_correlation_list.append(best_train_correlation)
    best_train_ev_list.append(best_train_ev)

    best_valid_mse_list.append(best_valid_loss)
    best_valid_r2_list.append(best_valid_r2)
    best_valid_correlation_list.append(best_valid_correlation)
    best_valid_ev_list.append(best_valid_ev)


    model.load_state_dict(torch.load(f'checkpoint_{job_id}_fold_{fold}.pth'), strict = False)  # Load best model for fold
    #testing 
    test_loss = 0
    actual_batch_values_test = []
    predicted_batch_values_test = []
    subject_ids_test = [] 
    gradients_list = []
    input_gradients_list = [] 

    # Get the subject directories from img_path to verify
    img_subject_ids = set([subject for subject in os.listdir(data.img_path) if subject.startswith("NDAR")])

    # with torch.no_grad():
    #     with autocast():
    for idx, (X, y) in enumerate(test_loader):
        X, y = X.to(device).float(), y.to(device).float()
        X = torch.unsqueeze(X, 1).float()
        X.requires_grad = True

        # Extract the src_subject_id for current batch
        batch_subject_ids = data.vars.index[data.test_idx[idx * batch_size:(idx + 1) * batch_size]]
        subject_ids_test.extend(batch_subject_ids)
        

        # Compare subject IDs with img_path directories
        mismatched_ids = [sub_id for sub_id in batch_subject_ids if sub_id not in img_subject_ids]
        if mismatched_ids:
            print(f"Warning: Mismatched subject IDs in test data: {mismatched_ids}")

        pred = model(X)
        pred = pred.squeeze()
        loss = criterion(pred, y)
        test_loss += (loss.item() * X.shape[0])

        # Append actual and predicted values to their lists
        if pred.dim() > 0:
            actual_batch_values_test.extend(y.cpu().numpy())
            predicted_batch_values_test.extend(pred.cpu().detach().numpy())

            model.zero_grad()  
            pred.sum().backward()  # Compute gradients
            gradients = X.grad.detach()
            input_gradients = X.detach() * gradients  
            input_gradients_np = input_gradients.cpu().numpy()
            gradients_np = gradients.cpu().numpy()


            gradients_list.append(gradients_np)
            input_gradients_list.append(input_gradients_np)

            gradients_list.append(gradients_np)
            input_gradients_list.append(input_gradients_np)

            # Check for NaN in actual and predicted values
            if np.isnan(np.sum(y.cpu().numpy())) or np.isnan(np.sum(pred.cpu().detach().numpy())):
                print(f"NaN detected in test set at batch {idx}")

    # Convert lists to numpy arrays for further processing
    actual_batch_values_test = np.array(actual_batch_values_test)
    predicted_batch_values_test = np.array(predicted_batch_values_test)

    # Check if actual and predicted values have aligned subject IDs
    aligned = np.array(subject_ids_test) == data.vars.index[data.test_idx]
    if not np.all(aligned):
        print("Subject IDs are not aligned between actual and predicted values in the test set!")
        # Print the mismatched indices for debugging
        mismatched_indices = np.where(aligned == False)
        print(f"Mismatched indices: {mismatched_indices}")

    avg_test_loss = test_loss / len(data.test_idx)
    test_metrics = calculate_metrics(actual_batch_values_test, predicted_batch_values_test)
    print(f"Average test loss: {avg_test_loss}")
    print(f"Epoch {e}/{epochs} - Average test loss: {avg_test_loss} - R2: {test_metrics['r2 score']:.4f}, EV: {test_metrics['ev']:.4f}, Correlation: {test_metrics['correlation']:.4f}")

    # Concatenate and save input-gradient products
    input_gradients_array = np.concatenate(input_gradients_list, axis=0)
    np.save(f"input_gradients_{job_id}_fold_{fold}.npy", input_gradients_array)
    print(f"Saved input-gradient products for fold {fold} to input_gradients_{job_id}_fold_{fold}.npy")

    # Store metrics for each fold
    mse_list.append(avg_test_loss)
    r2_list.append(test_metrics['r2 score'])
    correlation_list.append(test_metrics['correlation'])
    ev_list.append(test_metrics['ev'])
       

    
    print(f"Fold {fold} completed.")




# Calculate mean and standard deviation for the best training and validation metrics
mean_best_train_mse, std_best_train_mse = np.mean(best_train_mse_list), np.std(best_train_mse_list)
mean_best_train_r2, std_best_train_r2 = np.mean(best_train_r2_list), np.std(best_train_r2_list)
mean_best_train_correlation, std_best_train_correlation = np.mean(best_train_correlation_list), np.std(best_train_correlation_list)
mean_best_train_ev, std_best_train_ev = np.mean(best_train_ev_list), np.std(best_train_ev_list)

mean_best_valid_mse, std_best_valid_mse = np.mean(best_valid_mse_list), np.std(best_valid_mse_list)
mean_best_valid_r2, std_best_valid_r2 = np.mean(best_valid_r2_list), np.std(best_valid_r2_list)
mean_best_valid_correlation, std_best_valid_correlation = np.mean(best_valid_correlation_list), np.std(best_valid_correlation_list)
mean_best_valid_ev, std_best_valid_ev = np.mean(best_valid_ev_list), np.std(best_valid_ev_list)

# Calculate mean and standard deviation for test metrics
mean_mse, std_mse = np.mean(mse_list), np.std(mse_list)
mean_r2, std_r2 = np.mean(r2_list), np.std(r2_list)
mean_correlation, std_correlation = np.mean(correlation_list), np.std(correlation_list)
mean_ev, std_ev = np.mean(ev_list), np.std(ev_list)

# Print the results
print(f"Mean Best Training MSE: {mean_best_train_mse:.4f} +/- {std_best_train_mse:.4f}")
print(f"Mean Best Training R2: {mean_best_train_r2:.4f} +/- {std_best_train_r2:.4f}")
print(f"Mean Best Training Pearson Correlation: {mean_best_train_correlation:.4f} +/- {std_best_train_correlation:.4f}")
print(f"Mean Best Training Explained Variance: {mean_best_train_ev:.4f} +/- {std_best_train_ev:.4f}")

print(f"Mean Best Validation MSE: {mean_best_valid_mse:.4f} +/- {std_best_valid_mse:.4f}")
print(f"Mean Best Validation R2: {mean_best_valid_r2:.4f} +/- {std_best_valid_r2:.4f}")
print(f"Mean Best Validation Pearson Correlation: {mean_best_valid_correlation:.4f} +/- {std_best_valid_correlation:.4f}")
print(f"Mean Best Validation Explained Variance: {mean_best_valid_ev:.4f} +/- {std_best_valid_ev:.4f}")

print(f"Mean Test MSE: {mean_mse:.4f} +/- {std_mse:.4f}")
print(f"Mean Test R2: {mean_r2:.4f} +/- {std_r2:.4f}")
print(f"Mean Test Pearson Correlation: {mean_correlation:.4f} +/- {std_correlation:.4f}")
print(f"Mean Test Explained Variance: {mean_ev:.4f} +/- {std_ev:.4f}")

print('####################################################################')
print("Done")


print("End")