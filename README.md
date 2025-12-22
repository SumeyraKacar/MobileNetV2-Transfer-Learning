# 🌸 MobileNetV2 ile Çiçek Sınıflandırma (Transfer Learning)

Bu proje, PyTorch kütüphanesi ve **MobileNetV2** mimarisi kullanılarak doğadaki çiçek türlerini tanımak amacıyla geliştirilmiştir. 
**Transfer Learning** yöntemiyle eğitilen model, yerel ağırlıklar (.pth) üzerinden offline (internet gerektirmeden) çalışabilmektedir.

## 📝 Sunum Maddelerine Göre Proje Özeti

### 1. Problemin Tanımı ve Önemi
Modern derin öğrenme modellerini sıfırdan eğitmek devasa donanım ve zaman maliyeti gerektirir. Bu projede, kısıtlı imkanlarla yüksek başarı elde etmek için **Transfer Learning** yaklaşımı kullanılmıştır.

### 2. Kullanılan Veri Seti ve Analizi
* **Kaynak:** ImageNet (Genel) ve Kaggle Flowers Dataset (Özel).
* **İşlem:** Görüntüler model girişine uygun olarak 224x224 boyutuna getirilmiştir.
* **Normalizasyon:**ImageNet standartlarında (RGB kanalları için özel ortalama ve sapma değerleri) normalize edilerek modelin veriye daha hızlı uyum sağlaması sağlanmıştır.

### 3. Uygulanan Yöntem(ler)
* **Model:** MobileNetV2 (Hafif ve verimli mimari).
* **Yapı:** Önceden eğitilmiş katmanlar dondurulmuş (Frozen), sadece son sınıflandırma katmanı 5 çiçek türü için yeniden eğitilmiştir.
* **Teknik:** Depthwise Separable Convolution" yapısı sayesinde işlem yükü minimize edilmiştir.

### 4. Deneysel Sonuçlar ve Metrikler
* **Model Ağırlığı:** `my_model.pth` (Offline çalışmaya uygun).
* **Sonuç:** `test_resmi.jpg` üzerinde yapılan testlerde doğru sınıflandırma başarısı elde edilmiştir.

### 5. Genel Değerlendirme
Model, düşük işlem gücüyle dahi mobil cihazlarda çalışabilecek kadar hızlıdır. İyileştirme olarak veri artırımı (Data Augmentation) teknikleri eklenebilir.

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

## 🚀 Başarı Kanıtı
Yapılan testlerde model, daha önce hiç görmediği bir gül fotoğrafını (test_resmi.jpg) saniyeler içinde analiz etmiş ve şu sonucu üretmiştir:

YAPAY ZEKANIN TAHMİNİ: ===> ROSES <===
![Uploading image.png…]()


## 🧠 Teknik Özellikler
Model: MobileNetV2 (Hafif ve Verimli)

Yöntem: Transfer Learning (Önceden eğitilmiş ImageNet ağırlıkları kullanıldı)

Giriş: 224x224 RGB Görüntüler

Normalizasyon: ImageNet standartları (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])

Kütüphane: PyTorch & Torchvision
