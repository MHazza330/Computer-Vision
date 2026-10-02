import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def main():
    # 1. Cek Device (GPU / CUDA jika ada, atau CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Menggunakan device: {device}")

    # 2. Augmentasi & Transformasi Gambar (Sesuai Standar ImageNet & RET503)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'dataset'
    
    # Cek ketersediaan folder dataset
    if not os.path.exists(data_dir):
        print(f"[!] Folder '{data_dir}' tidak ditemukan. Pastikan Anda sudah menjalankan capture_dataset.py terlebih dahulu!")
        return

    # Load dataset menggunakan ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"[*] Kelas terdeteksi: {class_names}")
    print(f"[*] Jumlah data train: {dataset_sizes['train']} | val: {dataset_sizes['val']}")

    # 3. Load Transfer Learning Model: MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze feature extractor (backbone beku)
    for param in model.parameters():
        param.requires_grad = False

    # Ganti classifier head sesuai jumlah kelas (basket & bola_basket)
    num_classes = len(class_names)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimizer hanya memperbarui bobot classifier head baru
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    # 4. Training Loop
    num_epochs = 10
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n[+] Memulai proses pelatihan...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Simpan bobot terbaik berdasarkan akurasi validasi
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[+] Akurasi Validasi Terbaik: {best_acc:.4f}")
    
    # Load bobot terbaik dan simpan ke file .pth
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'humanoid_object_model.pth')
    print("[+] Model berhasil disimpan ke 'humanoid_object_model.pth'")

if __name__ == '__main__':
    main()