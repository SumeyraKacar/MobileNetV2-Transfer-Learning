# 🌸 MobileNetV2 ile Çiçek Sınıflandırma (Transfer Learning)

Bu proje, PyTorch kütüphanesi ve **MobileNetV2** mimarisi kullanılarak doğadaki çiçek türlerini tanımak amacıyla geliştirilmiştir. 
**Transfer Learning** yöntemiyle eğitilen model, internet bağlantısına ihtiyaç duymadan yerel ağırlıklar üzerinden çalışabilmektedir.

## 📂 Proje Yapısı
- **train.py**: Modelin eğitim sürecini yönetir. MobileNetV2 mimarisini yükler ve son katmanı bizim veri setimize göre günceller.
- **son_test.py**: Eğitilmiş `my_model.pth` dosyasını kullanarak tahminleme yapar.
- **my_model.pth**: Modelin tüm öğrenilmiş bilgilerini içeren ağırlık dosyası (Offline çalışma sağlar).
- **dataset/**: Modelin eğitimi için kullanılan görüntülerin bulunduğu klasör.

## 🛠️ Kurulum
Gerekli kütüphaneleri yüklemek için terminale şu komutu yazın:

    pip install -r requirements.txt

## 📊 Kullanım
Modeli Test Etme (Tahminleme)
Eğitilmiş modelimizi test etmek ve bir çiçeğin türünü öğrenmek için yani yerel bir resmi (test_resmi.jpg) sınıflandırmak şu komutu kullanıyoruz:

    python son_test.py

## 📈 Başarı Kanıtı
Yapılan testlerde model, daha önce hiç görmediği bir gül fotoğrafını (test_resmi.jpg) saniyeler içinde analiz etmiş ve şu sonucu üretmiştir:

YAPAY ZEKANIN TAHMİNİ: ===> ROSES <===

## 🧠 Teknik Özellikler
Model: MobileNetV2 (Hafif ve Verimli)

Yöntem: Transfer Learning (Önceden eğitilmiş ImageNet ağırlıkları kullanıldı)

Giriş: 224x224 RGB Görüntüler

Normalizasyon: ImageNet standartları (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])
