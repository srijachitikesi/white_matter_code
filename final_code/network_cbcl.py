#%%
import torch.nn as nn
import torch.nn.functional as F
import torch

#%%
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv3d(1, 8, 3, stride=1, padding=0) #(182,218,182)
        self.bn1 = nn.BatchNorm3d(8)
        self.cv2 = nn.Conv3d(8, 16, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(16)
        self.cv3 = nn.Conv3d(16, 32, 3,stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(32)
        self.cv4 = nn.Conv3d(32, 64, 3,stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(64)
        self.cv5 = nn.Conv3d(64, 64, 3,stride=1, padding=0)
        self.bn5 = nn.BatchNorm3d(64)
        
        self.pool = nn.MaxPool3d(2)
        # self.pool4 = nn.MaxPool3d(5)
        # self.fc1 = nn.Linear(2304, 1024)  # Adjusted to create hidden layer
        # self.fc2 = nn.Linear(1024, 128)     # New layer added
        # self.fc3 = nn.Linear(128,1)
        self.fc1 = nn.Linear(2304, 256)  # Adjusted to create hidden layer
        self.fc2 = nn.Linear(256, 1)     # New layer added
        # self.fc1 = nn.Linear(2304, 1)
        
        # self.fc1 = nn.Linear(9216, 4096)
        # self.fc2 = nn.Linear(4096, 1024)
        # self.fc3 = nn.Linear(1024,512)
        # self.fc4 = nn.Linear(512,1)

        # self.d3d = nn.Dropout3d(0.2)
        # self.dropout = nn.Dropout(0.7) 
        # self.hidden_dropout = nn.Dropout(0.5)
        self.hidden_dropout = nn.Dropout(0.3)

        
        self.layer2 = nn.Sequential(self.cv2,self.bn2,self.pool,nn.ReLU())
        self.layer3 = nn.Sequential(self.cv3,self.bn3,self.pool,nn.ReLU())
        self.layer4 = nn.Sequential(self.cv4,self.bn4,self.pool,nn.ReLU())
        self.layer1 = nn.Sequential(self.cv1,self.bn1,self.pool,nn.ReLU())
        self.layer5 = nn.Sequential(self.cv5,self.bn5,self.pool,nn.ReLU())


        self.convs = nn.Sequential(self.layer1,self.layer2,self.layer3,
                        self.layer4,self.layer5)
        # self.convs.apply(Network.init_weights)
        # self.dropout = nn.Dropout(0.7)

    def forward(self, img, data=None):
        
        img = self.convs(img)
        # img = self.global_pool(img)

        img = img.view(img.shape[0], -1)

        if data is not None:
            with torch.no_grad():
                img = torch.cat((img,torch.unsqueeze(data,1)),dim=1)
        
        # img = self.fc1(img)
        # img = self.dropout(img)
        # img = self.fc2(img)
        # img = self.dropout(F.relu(self.fc1(img)))

        img = F.relu(self.fc1(img))        # Hidden layer with ReLU activation
        img = self.hidden_dropout(img)     # Dropout after hidden layer
        # img = F.relu(self.fc2(img))        # Hidden layer with ReLU activation
        # img = self.hidden_dropout(img)  
        img = self.fc2(img)                # Final layer
        # img = self.dropout(img)            # Final dropout

        return img

    # @staticmethod
    # def init_weights(m):
    #     if isinstance(m, nn.Linear) or isinstance(m,nn.Conv3d):
    #         torch.nn.init.xavier_uniform_(m.weight)