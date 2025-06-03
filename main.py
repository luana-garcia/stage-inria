from train_unetrpp import MeteoUNetRPP
from train_unetrpp import MeteoDataset

import torch
from torch.utils.data import DataLoader

model = MeteoUNetRPP()
model.set_parameters()

data = MeteoDataset(channel=21)

data.show_time_frames()
data.show_channels()

# Dividir em treinamento e validação
train_size = int(0.8 * len(data))  # 80% para treinamento
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=model.batch_size, shuffle=False)

model.train(train_loader, val_loader)
model.save_model()