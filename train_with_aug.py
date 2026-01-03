import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image

# ==========================================
# 1. SİSTEM KONFİGÜRASYONU VE PARAMETRELER
# ==========================================
MOD = 'PREDICT'
AUG_DURUMU = 'T'

EPOCH_SAYISI = 30
BATCH_SIZE = 32
VERI_YOLU = 'dataset'

DURUM = "VAR" if AUG_DURUMU == 'T' else "YOK"
MODEL_DOSYA_ADI = f'model_master_aug_{DURUM}.pth'
GRAFIK_ANA = f'PROFESYONEL_ANALIZ_AUG_{DURUM}.png'
GRAFIK_HATA = f'HATA_ANALIZI_MATRISI_{DURUM}.png'
RAPOR_DOSYA = f'PERFORMANS_RAPORU_{DURUM}.txt'

# ==========================================
# 2. VERİ ÖN İŞLEME VE VERİ ARTIRIM STRATEJİLERİ
# ==========================================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ==========================================
# 3. MODEL EĞİTİMİ VE DOĞRULAMA DÖNGÜSÜ
# ==========================================
def train_model():
    print(f"\n[BİLGİ] Eğitim Süreci Başlatıldı: Durum={DURUM}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = datasets.ImageFolder(VERI_YOLU)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    if AUG_DURUMU == 'T':
        train_data.dataset.transform = train_transform_aug
    else:
        train_data.dataset.transform = val_transform

    val_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Transfer Learning: Pre-trained MobileNetV2 Kullanımı
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCH_SAYISI):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_accs.append(100. * correct / total)
        train_losses.append(r_loss / len(train_data))

        model.eval()
        v_loss, v_cor, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                v_tot += labels.size(0)
                v_cor += pred.eq(labels).sum().item()

        val_accs.append(100. * v_cor / v_tot)
        val_losses.append(v_loss / len(val_data))
        print(f"Epoch {epoch + 1}/{EPOCH_SAYISI} | Val Accuracy: %{val_accs[-1]:.1f}")

    # ==========================================
    # 4. PERFORMANS GRAFİKLERİNİN OLUŞTURULMASI
    # ==========================================
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Eğitim Başarımı')
    plt.plot(val_accs, label='Doğrulama Başarımı')
    plt.title('Eğitim Süreci Doğruluk Analizi')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(val_losses, label='Doğrulama Kaybı')
    plt.title('Eğitim Süreci Kayıp Analizi')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp (Loss)')
    plt.legend()
    plt.savefig(GRAFIK_ANA)
    plt.close()

    # ==========================================
    # 5. METRİK HESAPLAMA VE RAPORLAMA (F1-SCORE)
    # ==========================================
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report ve F1-Score Analizi
    report = classification_report(all_labels, all_preds, target_names=full_dataset.classes)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Raporun Dosyaya Yazılması (Unicode Desteği ile)
    with open(RAPOR_DOSYA, "w", encoding="utf-8") as f:
        f.write(f"MODEL PERFORMANS ANALİZ RAPORU (AUGMENTATION: {DURUM})\n" + "=" * 45 + "\n")
        f.write(report)
        f.write(f"\nAğırlıklı F1-Skoru (Weighted F1-Score): {f1:.4f}")

    # ==========================================
    # 6. KARMAŞIKLIK MATRİSİ GÖRSELLEŞTİRME
    # ==========================================
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.title(f'Karmaşıklık Matrisi Analizi (AUG: {DURUM})')
    plt.ylabel('Gerçek Sınıflar')
    plt.xlabel('Tahmin Edilen Sınıflar')
    plt.savefig(GRAFIK_HATA)
    plt.show()

    torch.save(model.state_dict(), MODEL_DOSYA_ADI)
    print(f"\n[BİLGİ] Performans metrikleri ve raporlar başarıyla oluşturuldu.")


# ==========================================
# 7. MODEL ÇIKARSAMA (PREDICTION) SİSTEMİ
# ==========================================
def predict_image(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(os.listdir(VERI_YOLU))

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))

    if os.path.exists(MODEL_DOSYA_ADI):
        model.load_state_dict(torch.load(MODEL_DOSYA_ADI, map_location=device))
        model = model.to(device).eval()

        img = Image.open(img_path).convert('RGB')
        img_t = val_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            confidence, pred = torch.max(prob, 0)

        plt.imshow(img)
        plt.title(f"Sınıf Tahmini: {classes[pred]} | Güven: %{confidence.item() * 100:.2f}")
        plt.axis('off')
        plt.show()
    else:
        print(f"[HATA] Model ağırlık dosyası bulunamadı. Lütfen önce eğitim gerçekleştirin.")


# ==========================================
# 8. ANA YÜRÜTÜCÜ BLOK
# ==========================================
if __name__ == '__main__':
    if MOD == 'TRAIN':
        train_model()
    elif MOD == 'PREDICT':
        # Örnek tahmin görseli yolu
        predict_image('External_Test/karahindiba1.jpg')