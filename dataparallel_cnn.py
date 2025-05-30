import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"  # disattiva infiniband
os.environ["NCCL_P2P_DISABLE"] = "1"  # disattiva P2P tra GPU

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
import json
from datetime import datetime, timedelta
import gc
import os 
import threading

print(f"Numero di CPU disponibili: {os.cpu_count()}")
print("🖥️ Device disponibile:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else "Solo CPU")

#--------------------------------------
# 1. STRUTTURA DATI E TRASFORMAZIONI
#--------------------------------------
class SaltAndPepperNoise:
    def __init__(self, prob=0.02):
        self.prob = prob

    def __call__(self, img):
        np_img = np.array(img)
        mask = np.random.choice([0, 1, 2], size=np_img.shape, p=[self.prob/2, 1-self.prob, self.prob/2])
        np_img[mask == 0] = 0    # Sale
        np_img[mask == 2] = 255  # Pepe
        return torch.tensor(np_img, dtype=torch.float32)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return img + torch.randn_like(img) * self.std + self.mean


class HDF5Dataset(Dataset):
    def __init__(self, h5_path, m_values, transform=None, apply_noise=False, num_threads=8):
        self.h5_path = h5_path
        self.m_values = m_values
        self.transform = transform
        self.apply_noise = apply_noise
        self.data = []  
        self.labels = []
        
        # Thread-local storage per file HDF5 (uno per worker)
        self._local = threading.local()

        # Parallelizza la lettura dei metadati
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
                    
        print(f"📊 Dataset caricato: {len(self.data)} immagini")

    def _process_image_metadata(self, m, img_id):
        return {"m": m, "id": img_id}, self.m_values.index(m)

    def _get_h5_file(self):
        """Ottieni file HDF5 thread-local (uno per DataLoader worker)"""
        if not hasattr(self._local, 'h5_file'):
            self._local.h5_file = h5py.File(self.h5_path, 'r')
        return self._local.h5_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        m, img_id = item["m"], item["id"]

        h5f = self._get_h5_file()
        
        try:
            img_group = h5f[f"images/m_{m}"][img_id]
            
            # ✅ LETTURA DIRETTA come array NumPy
            image = np.stack([
                img_group["CHM_direct"][()],
                img_group["CCM_direct"][()],
                img_group["CCM_reciprocal"][()],
            ], axis=0)
            
        except KeyError as e:
            print(f"Errore lettura immagine m={m}, id={img_id}: {e}")
            # Fallback: immagine zero
            image = np.zeros((3, 512, 512), dtype=np.float32)

        
        image = torch.from_numpy(image.astype(np.float32))
        
        
        if self.apply_noise:
            noise_std = 0.01 * torch.rand(1).item()  # Random noise level
            image += torch.randn_like(image) * noise_std
        
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

    def __del__(self):
        """Cleanup quando il dataset viene distrutto"""
        if hasattr(self._local, 'h5_file'):
            try:
                self._local.h5_file.close()
            except:
                pass

def selective_transform(image):
        """Applica Gaussian Noise solo al canale CCM_reciprocal (indice 2)."""
        noise_channel = AddGaussianNoise(mean=0.0, std=0.05)(image[2].unsqueeze(0))  
        return torch.cat([image[:2], noise_channel], dim=0)

#--------------------------------------
#2. MODEL
#--------------------------------------

class FlexibleCNN(nn.Module):
    def __init__(self, num_classes, channels, input_size=(3, 512, 512)):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        # Calcolo dimensioni in base a dummy input
        input_features = self._get_classifier_input_features(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )

    def _get_classifier_input_features(self, input_size):
        # input_size: (C, H, W)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)  
            x = self.features(dummy_input)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  
        return x                 

#--------------------------------------
# 3. UTILITIES PER IL TRAINING
#--------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth.tar"):
    """Modificata per gestire DataParallel"""
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvato in {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """Modificata per gestire DataParallel"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        
        # Carica lo state_dict nel modello base (prima di DataParallel)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint caricato da {filename} (epoca {epoch})")
        return epoch, loss
    else:
        print(f"Nessun checkpoint trovato in {filename}")
        return 0, None

class TrainingLogger:
    """Registra metriche e parametri del training"""
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.data = {
            "params": {},
            "metrics": {
                "train_loss": [],
                "val_acc": [],
                "val_f1": [],
                "val_mae": [],
                "val_mse": [],
                "best_val_acc": 0,
                "best_val_f1": 0
            }
        }
    
    def log_params(self, params):
        self.data["params"].update(params)
    
    def log_metrics(self, train_loss, val_acc, val_f1=None, val_mae=None, val_mse=None):
        self.data["metrics"]["train_loss"].append(float(train_loss))
        self.data["metrics"]["val_acc"].append(float(val_acc))
        self.data["metrics"]["val_f1"].append(float(val_f1))
        self.data["metrics"]["val_mae"].append(float(val_mae))
        self.data["metrics"]["val_mse"].append(float(val_mse))

        if val_acc > self.data["metrics"]["best_val_acc"]:
            self.data["metrics"]["best_val_acc"] = float(val_acc)
        if val_f1 is not None:
            self.data.setdefault("metrics", {}).setdefault("val_f1", []).append(float(val_f1))
        if val_mae is not None:
            self.data.setdefault("metrics", {}).setdefault("val_mae", []).append(float(val_mae))
        if val_mse is not None:
            self.data.setdefault("metrics", {}).setdefault("val_mse", []).append(float(val_mse))
        self._save()
    
    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=4)

#--------------------------------------
# 4. TRAINING
#--------------------------------------

def main():
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
    
    # Configurazione device, logger, m_values
    
    logger = TrainingLogger(args.output_dir)
    logger.log_params(vars(args))
    m_values = [2, 3, 4]

    
    # Dataset e split
    full_dataset = HDF5Dataset(args.h5_path, m_values)
    y = y = full_dataset.labels
    train_val_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.1, stratify=y, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, stratify=[y[i] for i in train_val_idx], random_state=42)


    # DataLoader
    train_dataset = HDF5Dataset(args.h5_path, m_values, transform=transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        SaltAndPepperNoise(prob=0.02),
    ]), apply_noise=True)
    
    test_dataset = HDF5Dataset(args.h5_path, m_values)
    
    train_loader = DataLoader(Subset(train_dataset, train_idx), 
                            batch_size=args.batch_size * torch.cuda.device_count(), 
                            shuffle=True, 
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2)
    
    val_loader = DataLoader(Subset(test_dataset, val_idx), 
                          batch_size=args.batch_size * torch.cuda.device_count(),
                          num_workers=8,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=2)

    # Modello e ottimizzatore
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    model = FlexibleCNN(num_classes=len(m_values), channels=args.channels)

  
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")

        # === LOAD CHECKPOINT BEFORE MOVING TO GPU ===
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"📦 Caricamento checkpoint da {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')  # Carica su CPU prima
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        if args.load_checkpoint:
            print(f"⚠️ Checkpoint non trovato: {args.checkpoint_path}. Si parte da zero.")
        else:
            print("ℹ️ Addestramento da zero (no checkpoint richiesto).")

    # === MOVE TO GPU AFTER LOADING ===
    model = model.to(device)

    # === APPLY DATAPARALLEL IF NEEDED ===
    if torch.cuda.device_count() > 1:
        print(f"🧠 Usando {torch.cuda.device_count()} GPU con DataParallel")
        model = nn.DataParallel(model)

    # === OPTIMIZER AFTER DATAPARALLEL ===
    optimizer = optim.Adam(model.parameters(), lr=args.lr * torch.cuda.device_count())  # Scala il LR
    criterion = nn.CrossEntropyLoss()
    #use_amp = False  # o leggilo da argparse
    scaler = GradScaler(enabled=args.use_amp)

    print(f"Model parameters are on device: {next(model.parameters()).device}")

    # Se usi DataParallel, puoi accedere al modulo originale con model.module
    if hasattr(model, 'module'):
        print(f"Original model device: {next(model.module.parameters()).device}")    
    



    best_val_acc = 0.0  
    best_val_f1 = 0
    print("Inizio training...", flush=True)

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch} iniziato", flush=True)
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        # for batch_idx, (images, labels) in enumerate(train_loader):
            #print(f"Batch {batch_idx} caricato", flush=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        # Validazione
        model.eval()
        val_correct = 0
        val_total = 0
        all_y_true, all_y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                # val_correct += predicted.eq(labels).sum().item()
                # val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(predicted.cpu().numpy())

        # Calcola metriche (solo su GPU 0)
        if device == torch.device('cuda:0'):
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            val_f1 = f1_score(all_y_true, all_y_pred, average='macro')
            val_mae = mean_absolute_error(all_y_true, all_y_pred)
            val_mse = mean_squared_error(all_y_true, all_y_pred)

            print(f"Epoch {epoch+1}:")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

            # Salva checkpoint periodico 
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model.module if hasattr(model, 'module') else model,
                optimizer,
                epoch,
                train_loss,
                filename=os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth.tar")
            )

        # Salva il modello migliore (pesi completi)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Salva solo i pesi del modello
            torch.save(
                model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                os.path.join(args.output_dir, "best_model_weights.pth")
            )
            # Salva checkpoint completo (modello+ottimizzatore)
            save_checkpoint(
                model.module if hasattr(model, 'module') else model,
                optimizer,
                epoch,
                train_loss,
                filename=os.path.join(args.output_dir, "best_model_checkpoint.pth.tar")
            )

        # Logging
        logger.log_metrics(
            train_loss=train_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            val_mae=val_mae,
            val_mse=val_mse
        )

if __name__ == "__main__":
    main()