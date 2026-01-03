🌸 Flower Classification with MobileNetV2 & Transfer Learning

Bu proje, MobileNetV2 mimarisini kullanarak çiçek türlerini (Papatya, Karahindiba, Gül, Ayçiçeği, Lale) yüksek doğrulukla sınıflandırmayı amaçlayan bir Derin Öğrenme çalışmasıdır.

Projenin odak noktası, Transfer Learning (Transfer Öğrenme) ve Data Augmentation (Veri Artırma) tekniklerinin modelin genelleme yeteneği üzerindeki etkisini analiz etmektir.

🚀 Öne Çıkan Özellikler
Mimari: Google MobileNetV2 (Hafif ve verimli mobil uyumlu mimari).

Teknik: Transfer Learning (ImageNet ağırlıkları ile ön eğitimli).

Dondurma (Freezing): Feature Extractor katmanları dondurulmuş, sadece Classifier eğitilmiştir.

Kıyaslama: Veri artırma (Augmentation) uygulanan ve uygulanmayan modellerin başarım karşılaştırması.

Analiz: Eğitim sonrası otomatik "Classification Report" ve "Confusion Matrix" üretimi.

📊 Deneysel Sonuçlar

Yapılan testler sonucunda Veri Artırma (Augmentation) yönteminin Overfitting (Aşırı Öğrenme) problemini nasıl engellediği somut bir şekilde gözlemlenmiştir:

Metrik                 Veri Artırma YOK             Veri Artırma VAR 
Test Accuracy          %91 (Ezberleme Eğilimli)     %89 (Daha Güvenilir)
Ağırlıklı F1-Skoru     0.9131                       0.8870
Eğitim/Val Farkı       Yüksek                       Düşük (Stabil)

Analiz Notu: Augmentation uygulanmayan model test setinde yüksek puan alsa da, eğitim kaybının (loss) sıfıra çok yakın olması modelin resimleri ezberlediğini göstermektedir. 
Augmentation uygulanan model ise gerçek dünya verilerine karşı çok daha dayanıklıdır.

🛠️ Kurulum
1. Bu depoyu klonlayın:

        git clone https://github.com/SumeyraKacar/MobileNetV2-Flower-Classification.git
        cd MobileNetV2-Flower-Classification

3. Gerekli kütüphaneleri yükleyin:

        pip install -r requirements.txt

💻 Kullanım

Modeli Eğitmek İçin:
train_with_aug.py dosyasındaki MOD değişkenini 'TRAIN' olarak ayarlayın ve çalıştırın:

MOD = 'TRAIN'
AUG_DURUMU = 'T' # Veri artırmayı aktif eder

Tahmin Yapmak İçin:
Eğitilmiş model üzerinden bir görseli test etmek için MOD değişkenini 'PREDICT' yapın:

MOD = 'PREDICT'
predict_image('test_gorsel.jpg')

📁 Proje Yapısı

dataset/: Çiçek resimlerinin bulunduğu eğitim ve test klasörleri.

train_with_aug.py: Ana eğitim ve tahmin scripti.

model_master_aug_VAR.pth: Eğitilmiş model ağırlıkları.

PROFESYONEL_ANALIZ_...png: Eğitim süreci başarı ve kayıp grafikleri.

requirements.txt: Gerekli Python bağımlılıkları.

🎓 Akademik Referans

Bu çalışma Ankara Bilim Üniversitesi - Yapay Zeka Mühendisliği (AIE 521) kapsamında geliştirilmiştir.

Geliştirici: Sümeyra KACAR

Danışman: Dr. Öğretim Üyesi Abdülhalik OĞUZ


⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
