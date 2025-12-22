import torch
from torchvision import models, transforms
from PIL import Image
import os

# 1. Klasöründeki sınıf isimleri
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulip']

# 2. Modeli internete hiç sormadan bilgisayarında kur
model = models.mobilenet_v2(weights=None) 
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(classes))

# 3. Senin az önce oluşturduğun 'my_model.pth' dosyasını yükle
# Bu satır internete asla bağlanmaz
model.load_state_dict(torch.load('my_model.pth', map_location=torch.device('cpu')))
model.eval()

# 4. Resim Hazırlama Ayarları
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def tahmin_et(resim_yolu):
    img = Image.open(resim_yolu).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        out = model(batch_t)
    _, index = torch.max(out, 1)
    return classes[index[0]]

# 5. Çalıştır
resim_adi = 'test_resmi.jpg'
if os.path.exists(resim_adi):
    sonuc = tahmin_et(resim_adi)
    print("\n" + "="*40)
    print(f"YAPAY ZEKANIN TAHMİNİ: {sonuc.upper()}")
    print("="*40)
else:

    print(f"\nHata: {resim_adi} bulunamadı!")
