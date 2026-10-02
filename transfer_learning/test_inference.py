import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Konfigurasi Device (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Menggunakan device: {device}")

# 2. Definisikan Kelas (Sesuaikan urutan abjad folder dataset: 'basket' dan 'bola')
class_names = ['basket', 'bola']  # Folder disort secara alfabetis oleh torchvision

# 3. Inisialisasi Model & Load Weight (.pth)
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Load checkpoint model hasil training
model_path = 'humanoid_object_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 4. Pipeline Preprocessing Gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Buka Webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("[!] Gagal membuka webcam.")
    exit()

print("[*] Mengakses webcam... Tekan 'q' pada jendela video untuk keluar.")

# Batas minimal keyakinan model untuk menganggap objek ada
CONFIDENCE_THRESHOLD = 70.0  # %

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Gagal membaca frame dari webcam.")
        break

    # Konversi OpenCV Frame (BGR) ke PIL Image (RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Preprocessing & Tambah Batch Dimension
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediksi
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Ambil persentase masing-masing kelas secara terpisah
    basket_prob = probabilities[0].item() * 100
    bola_prob = probabilities[1].item() * 100

    # Logika Penentuan Objek Terdeteksi
    if bola_prob >= CONFIDENCE_THRESHOLD and bola_prob > basket_prob:
        text = f"BOLA: {bola_prob:.1f}%"
        color = (0, 255, 0)  # Hijau
    elif basket_prob >= CONFIDENCE_THRESHOLD and basket_prob > bola_prob:
        text = f"BASKET: {basket_prob:.1f}%"
        color = (0, 165, 255)  # Oranye
    else:
        text = "OBJEK TIDAK DITEMUKAN"
        color = (0, 0, 255)  # Merah

    # Tampilkan Hasil di Layar Video
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("FIRA Real-time Test Inference", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()