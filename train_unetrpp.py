import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import glob
import plotly.io as pio
pio.renderers.default = "vscode"
import os

from mfai.mfai.pytorch.models import unetrpp
import torch.nn as nn
from tqdm import tqdm

class MeteoDataset(Dataset):
    def __init__(self, channel, dir = None, transform=None):
        # Retrieving the data
        if dir == None:
            dirdata = "../scratch/shared/Titan/data_laurent/2023-01-0/"
        else:
            dirdata = dir

        self.subdirdata=glob.glob(dirdata+"*")
        self.subdirdata.sort()

        self.possible_channels = [os.path.basename(x) for x in glob.glob(self.subdirdata[0]+"/*")]
        self.possible_channels.sort()

        self.transform = transform

        # extend to multichannel
        # input_img = np.stack([self.meteo_cnn.getting_data_by_channel(c)[idx] for c in channels], axis=0)

        self.getting_data_by_channel(channel)

    # Return number of pairs of training (t, t+1)
    def __len__(self):
        return len(self.data) - 1

    # Convert pairs of data to Tensors
    def __getitem__(self, idx):
        # Input: image in t
        input_img = self.data[idx, :, :]
        # Output: image in t+1
        target_img = self.data[idx + 1, :, :]
        
        # Converter para tensores e adicionar dimensão de canal
        input_img = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0)  # (1, 512, 640)
        target_img = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)  # (1, 512, 640)

        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)

        return input_img, target_img
    
    def show_time_frames(self):
        # Listing every directory of time (each one with the multiple channels)
        print("Possible time frames:", self.subdirdata)

    def show_channels(self):
        # Listing every directory for each channel of weather information
        print("\nPossible channels:")
        for i in range(len(self.possible_channels)):
            print(i,":",self.possible_channels[i])
    
    # Getting the images by a especific channel
    def getting_data_by_channel(self, channel):
        obs1=np.load(self.subdirdata[0]+"/"+self.possible_channels[channel])

        stacked_images=np.zeros([len(self.subdirdata),obs1.shape[0],obs1.shape[1]])

        for i in range(len(self.subdirdata)):
            stacked_images[i,:,:]=np.load(self.subdirdata[i]+"/"+self.possible_channels[channel])

        # Normalize data (scale for [0, 1])
        self.data = (stacked_images - stacked_images.min()) / (stacked_images.max() - stacked_images.min())
        
        return self.data
    
    def show_data_by_channel(self, stacked_images, start = 0, end = 10):
        print("stacked_images shape:",stacked_images.shape)
        print("stacked_images min:",stacked_images.min())
        print("stacked_images max:",stacked_images.max())

        for i in range(start, end):
            plt.matshow(stacked_images[i,:,:])
            plt.show()

class MeteoUNetRPP:
    def __init__(self, batch_size = 4, num_epochs = 50):
        # Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = 1e-4
        # Model instantiation
        self.model=unetrpp.UNetRPP(in_channels=1,out_channels=1,input_shape=[512, 640]).to(self.device)
    
    # Loss and optimization functions
    def set_parameters(self):
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
    
    def train(self):
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}")
            
            # Validação
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    val_loss += self.criterion(outputs, targets).item()
            
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Atualizar scheduler
            self.scheduler.step(val_loss)
    
    def save_model(self):
        # Salvar o modelo treinado
        torch.save(self.model.state_dict(), "unetrpp_meteo_canal21.pth")

############## Execute training ############## 



# model = MeteoUNetRPP()
# model.set_parameters()

# data = MeteoDataset(channel=21)

# # Dividir em treinamento e validação
# train_size = int(0.8 * len(data))  # 80% para treinamento
# val_size = len(data) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=model.batch_size, shuffle=False)

# model.train()
# model.save_model()