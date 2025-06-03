import torch
import matplotlib.pyplot as plt
from train_unetrpp import MeteoUNetRPP
from train_unetrpp import MeteoDataset
from torch.utils.data import DataLoader

# Inicializar MeteoCNN e dataset
meteo = MeteoUNetRPP()
dataset = MeteoDataset(channel=21)

val_loader = DataLoader(dataset, batch_size=meteo.batch_size, shuffle=False)

# Carregar modelo treinado
model = meteo.model
model.load_state_dict(torch.load("unetrpp_meteo_canal21.pth"))
model.eval()

# Avaliar
criterion = torch.nn.MSELoss()
val_loss = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(meteo.device), targets.to(meteo.device)
        outputs = model(inputs)
        val_loss += criterion(outputs, targets).item()
        
        # Visualizar algumas previsões
        for i in range(min(2, inputs.size(0))):  # Mostrar 2 exemplos
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Input")
            plt.imshow(inputs[i, 0].cpu().numpy(), cmap='viridis')
            plt.subplot(1, 3, 2)
            plt.title("Target")
            plt.imshow(targets[i, 0].cpu().numpy(), cmap='viridis')
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(outputs[i, 0].cpu().numpy(), cmap='viridis')
            plt.show()

val_loss /= len(val_loader)
print(f"Validation MSE: {val_loss:.6f}")