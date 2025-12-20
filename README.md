# MobileNetV2 ile Görüntü Sınıflandırma ve Transfer Learning

Bu proje, Google'ın **MobileNetV2** mimarisini kullanarak nesne tanıma yapmayı ve **Transfer Learning** (Transfer Öğrenme) mantığını teknik olarak göstermeyi amaçlar.

## 📁 Proje Klasör Yapısı
- **model.py**: TensorFlow Hub üzerinden MobileNetV2 modelini yükler. [cite_start]`is_transfer` parametresi ile modelin iki farklı modda çalışmasını sağlar[cite: 2].
- **test.py**: Klasördeki `daisy.jpg` resmini kullanarak genel nesne tanıma testi yapar[cite: 5, 2].
- **train.py**: Modelin son katmanının (top layer) 5 sınıf için nasıl özelleştirildiğini (Transfer Learning) simüle eder.
- **ImageNetLabels.txt**: Modelin tanıdığı 1000 farklı nesnenin metin etiketlerini içerir.

## 🛠️ Kurulum
Projenin çalışması için gerekli kütüphaneleri şu komutla yükleyebilirsiniz:
```bash
pip install -r requirements.txt

## 📊 Kullanım

## 1. Tahminleme Yapmak (Demo)
Modelin hazır bilgilerini kullanarak yerel bir resmi sınıflandırmak için:

python test.py

## 2. Transfer Learning Mimarisini İncelemek
Modelin son katmanının (Dense Layer) nasıl değiştiğini görmek için:

python train.py
