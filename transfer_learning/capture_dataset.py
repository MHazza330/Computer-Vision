import cv2
import os
import random

# Direktori tujuan
BASE_DIR = 'dataset'
CLASSES = ['bola_basket', 'basket']

# Buat direktori jika belum ada
for cls in CLASSES:
    os.makedirs(os.path.join(BASE_DIR, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'val', cls), exist_ok=True)

# Inisialisasi Kamera (0 untuk webcam bawaan)
cap = cv2.VideoCapture(1)

count_bola = 0
count_basket = 0

print("=" * 50)
print("   PROGRAM CAPTURE DATASET FIRA HUROCUP BASKETBALL")
print("=" * 50)
print(" Petunjuk Penggunaan:")
print("  - Tekan 'b' : Ambil foto BOLA BASKET")
print("  - Tekan 'r' : Ambil foto RING BASKET")
print("  - Tekan 'q' : Keluar dari program")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Gagal mengakses kamera.")
        break

    # Tampilkan petunjuk dan status jumlah foto pada layar kamera
    display_frame = frame.copy()
    info_text1 = f"[b] Bola Basket: {count_bola}"
    info_text2 = f"[r] Ring Basket : {count_basket}"
    
    cv2.putText(display_frame, info_text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, info_text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, "Tekan 'q' untuk keluar", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Capture Dataset - Press 'b' (Bola) or 'r' (Ring)", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # Logika Simpan Foto Bola Basket (Tekan 'b')
    if key == ord('b'):
        split = 'train' if random.random() < 0.8 else 'val'
        filename = f"bola_{count_bola:04d}.jpg"
        filepath = os.path.join(BASE_DIR, split, 'bola_basket', filename)
        
        cv2.imwrite(filepath, frame)
        count_bola += 1
        print(f"[+] Foto BOLA disimpan ke: {filepath}")

    # Logika Simpan Foto Ring Basket (Tekan 'r')
    elif key == ord('r'):
        split = 'train' if random.random() < 0.8 else 'val'
        filename = f"basket_{count_basket:04d}.jpg"
        filepath = os.path.join(BASE_DIR, split, 'basket', filename)
        
        cv2.imwrite(filepath, frame)
        count_basket += 1
        print(f"[+] Foto RING disimpan ke: {filepath}")

    # Keluar dari Program (Tekan 'q')
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[*] Total foto terkumpul -> Bola: {count_bola} | Ring: {count_basket}")