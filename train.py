import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Veri Hazırlığı
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Senin hazırladığın 'dataset' klasörünü okuyoruz
dataset = datasets.ImageFolder(root='dataset', transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. Model: MobileNetV2 (Raporundaki mimari)
model = models.mobilenet_v2(weights='DEFAULT')
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Eğitim Başlıyor
print(f"Eğitim başlıyor... Sınıflar: {dataset.classes}")
model.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(train_loader):.4f}")

# 4. Modeli Kaydet (.pth formatında)
torch.save(model.state_dict(), 'my_model.pth')
print("\nBAŞARILI! 'my_model.pth' dosyası oluşturuldu.")s