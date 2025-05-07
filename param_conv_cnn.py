import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import h5py
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
import os 

print(f"Numero di CPU disponibili: {os.cpu_count()}")
print("🖥️ Device disponibile:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Solo CPU")

# Assicurati di definire queste classi se non sono già presenti
class SaltAndPepperNoise:
    def __init__(self, prob=0.02):
        self.prob = prob

    def __call__(self, img):
        # Converte in numpy per manipolare i pixel
        np_img = np.array(img)
        mask = np.random.choice([0, 1, 2], size=np_img.shape, p=[self.prob/2, 1 - self.prob, self.prob/2])
        np_img[mask == 0] = 0    # Sale (bianco)
        np_img[mask == 2] = 255  # Pepe (nero)
        return torch.tensor(np_img, dtype=torch.float32)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return img + torch.randn_like(img) * self.std + self.mean
    

# Funzioni per i checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvato in {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint caricato da {filename} (epoca {epoch})")
        return epoch, loss
    else:
        print(f"Nessun checkpoint trovato in {filename}")
        return 0, None

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, m_values, transform=None, apply_noise=False, num_threads=8):
        self.h5_path = h5_path
        self.m_values = m_values
        self.transform = transform
        self.apply_noise = apply_noise  # 🔥 Opzione per applicare noise direttamente
        self.data = []  
        self.labels = [] 

        # Parallelizza la lettura del file HDF5
        with h5py.File(h5_path, "r") as h5f:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for m in m_values:
                    m_group = h5f[f"images/m_{m}"]
                    for img_id in m_group:
                        futures.append(executor.submit(self._process_image_metadata, m, img_id))

                for future in futures:
                    data_entry, label = future.result()
                    self.data.append(data_entry)
                    self.labels.append(label)

    def _process_image_metadata(self, m, img_id):
        return {"m": m, "id": img_id}, self.m_values.index(m)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        m, img_id = item["m"], item["id"]

        with h5py.File(self.h5_path, "r") as h5f:
            img_group = h5f[f"images/m_{m}"][img_id]
            image = np.stack([
                img_group["CHM_direct"][()],
                img_group["CCM_direct"][()],
                img_group["CCM_reciprocal"][()],
            ], axis=0)

        image = torch.tensor(image, dtype=torch.float32)

        # **🔥 Applichiamo trasformazioni solo quando richiesto**
        if self.transform:
            image = self.transform(image)
        if self.apply_noise:  
            image = selective_transform(image)  

        label = self.m_values.index(m)  
        return image, label

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()

        # 🔹 Primo strato convoluzionale: 3 canali → 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalizzazione batch
        self.dropout1 = nn.Dropout(0.3)  # Dropout al 30%

        # 🔹 Secondo strato convoluzionale: 32 → 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.3)

        # 🔹 Pooling layer per ridurre la dimensione spaziale
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 🔹 Fully Connected Layers
        self.fc1 = nn.Linear(64 * 128 * 128, 128)  # Supponendo input 512x512
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x / 255.0  # Normalizzazione tra 0 e 1
        
        # 🔹 Conv 1 → BN → ReLU → Pool → Dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        # 🔹 Conv 2 → BN → ReLU → Pool → Dropout
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        # 🔹 Flatten
        x = x.view(x.size(0), -1)

        # 🔹 Fully Connected 1 → BN → ReLU → Dropout
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)

        # 🔹 Fully Connected 2 (Output)
        x = self.fc2(x)

        return x
    


# 🔽 Parser per argomenti da riga di comando
parser = argparse.ArgumentParser(description='CNN Training con parametri variabili')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=25, help='Numero di epoche')
parser.add_argument('--channels', type=int, default=64, help='Numero di canali conv')
parser.add_argument('--h5_path', type=str, required=True, help='Percorso al file HDF5')
parser.add_argument('--output_dir', type=str, default="./checkpoints", help='Directory per i checkpoint')
parser.add_argument('--checkpoint_path', type=str, default="checkpoint_final.pth.tar", help='Percorso del checkpoint')
parser.add_argument('--load_checkpoint', action='store_true', help='Carica il checkpoint esistente per riprendere il training')
parser.add_argument("--use_amp", action="store_true", help="Abilita Automatic Mixed Precision (AMP)")
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)


# 🔽 Parametri generali
m_values = [2, 3, 4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔽 Dataset e split train/val/test
full_dataset = HDF5Dataset(h5_path=args.h5_path, m_values=m_values)
y = full_dataset.labels
train_val_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.1, stratify=y, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, stratify=[y[i] for i in train_val_idx], random_state=42)

def selective_transform(image):
    """Applica Gaussian Noise solo al canale CCM_reciprocal (indice 2)."""
    noise_channel = AddGaussianNoise(mean=0.0, std=0.05)(image[2].unsqueeze(0))  # Aggiungiamo batch dim
    return torch.cat([image[:2], noise_channel], dim=0)  # Ricombiniamo i canali correttamente

train_dataset = HDF5Dataset(args.h5_path, m_values, transform=transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    SaltAndPepperNoise(prob=0.02),
]), apply_noise=True)

test_dataset = HDF5Dataset(args.h5_path, m_values)


train_loader = DataLoader(torch.utils.data.Subset(train_dataset, train_idx), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)
val_loader = DataLoader(torch.utils.data.Subset(test_dataset, val_idx), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=2)
test_loader = DataLoader(torch.utils.data.Subset(test_dataset, test_idx), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=2)

# 🔽 Modello con canali variabili
class FlexibleCNN(nn.Module):
    def __init__(self, num_classes, channels):
        super(FlexibleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear((channels * 2) * 128 * 128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x / 255.0
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.bn3(self.fc1(x))))
        return self.fc2(x)

model = FlexibleCNN(num_classes=len(m_values), channels=args.channels).to(device)
checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


# ⚙️ Scaler per AMP
scaler = GradScaler(enabled=args.use_amp)

# 📂 Caricamento se richiesto
start_epoch = 0
if args.load_checkpoint and os.path.exists(checkpoint_path):
    print(f"📂 Caricamento del checkpoint da {checkpoint_path}")
    start_epoch, _ = load_checkpoint(model, optimizer, filename=checkpoint_path)
    print(f"✅ Checkpoint caricato. Riprendo da epoca {start_epoch+1}")
else:
    start_epoch = 0

best_val_acc = 0.0  # Per salvare solo il modello migliore

# 🔽 Training loop
for epoch in range(start_epoch, args.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward con AMP se attivo
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100. * correct / total
    print(f"\nTrain Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    # 🔍 Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=args.use_amp):
                outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total
    print(f"Val Acc: {val_acc:.2f}%")

    # 💾 Salva il modello se migliora in validazione
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, optimizer, args.epochs, loss=best_val_acc, filename=os.path.join(args.output_dir, "final_model.pth.tar"))
        print(f"💾 Nuovo modello salvato con Val Acc: {val_acc:.2f}%")

# 🔽 Salvataggio finale
os.makedirs(args.output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))