import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit


class CustDataset(Dataset):
    
    def __init__(self, img_path='/data/neuromark2/Data/ABCD/DTI_Data_BIDS/Raw_Data/',
                 label_file='matching_subjects_CBCL_final.csv', transform=None,
                 target_transform=None, train=True, valid=False, random_state=52):
        path = '/data/users1/schitikesi1/CBCL/data/'
        print("Initializing CustomDataset 1")

        self.img_path = img_path
        self.dirs = os.listdir(img_path)
        cerebellum_frontal_mask_path = '/data/users1/schitikesi1/Myresearch/mask.nii'
        cerebellum_frontal_parietal_mask_path = '/data/users1/schitikesi1/Myresearch/parietal_mask.nii'
        frontal_thalamus_parietal_mask_path = '/data/users1/schitikesi1/Myresearch/frontal_thalamus_parietal_mask.nii'

        # self.mask = nib.load(mask_path).get_fdata()
        self.mask = nib.load(cerebellum_frontal_mask_path).get_fdata()
  
        row_values = 9177

        self.vars = pd.read_csv(path + label_file, index_col='src_subject_id',
                                usecols=['src_subject_id', 'cbcl_scr_syn_attention_r'], nrows=row_values)       
        self.num_selected_rows = len(self.vars)  # Store the length of selected rows
        
        # Print the number of selected rows
        print("Number of rows selected:", self.num_selected_rows)
        
        self.vars.columns = ['cbcl_scr_syn_attention_r']

        data_to_scale = np.array(self.vars['cbcl_scr_syn_attention_r']).reshape(-1,1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        self.vars['new_score'] = scaled_data.ravel()

        # self.vars['new_score'] = self.vars['tfmri_nb_all_beh_c2b_rate']

        print("Loaded labels...")

        sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        print("sss done")
        # self.train_idx, self.test_idx = next(sss.split(np.zeros_like(self.vars),
        #                                            self.vars.new_score.values))
        self.train_idx = list(range(int(0.8 * row_values)))
        self.test_idx = list(range(int(0.8 * row_values), row_values))

        # if train or valid:
        #     self.vars = self.vars.iloc[train_idx]
        # else:
        #     test_vars = self.vars.iloc[self.test_idx]
        
        self.vars = self.vars.sort_index()   
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        print("CustomDataset initialized.")

    
    def __len__(self):
        return len(self.vars)

    def __getitem__(self, idx):
        subject_dir = os.path.join(self.img_path, self.vars.index[idx])
        # Check if the subject directory exists
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Subject directory {subject_dir} not found.")
        
        # Specify the path to the target file
        target_file_path = os.path.join(subject_dir, 'Baseline', 'dti', 'dti_FA', 'tbdti32ch_FA.nii.gz')
        
        # Check if the target file exists
        if not os.path.exists(target_file_path):
            raise FileNotFoundError(f"Target file {target_file_path} not found.")
        
        # Load the data
        img = nib.load(target_file_path).get_fdata()


        label = self.vars.iloc[idx]

        # Preprocess image data
        # img = torch.tensor(img)
        # img = (img - img.mean()) / img.std()
        img = torch.tensor(img)
        img = (img - img.mean()) 
        img = img * self.mask #applying mask
        if torch.sum(torch.isnan(img)) > 0:
            print(f'Custom dataset, {idx}')
            exit(-1)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label['new_score']
