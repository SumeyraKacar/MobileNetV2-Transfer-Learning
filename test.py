import numpy as np
import PIL.Image as Image
from model import get_model

# Etiketleri oku
with open("ImageNetLabels.txt", "r") as f:
    labels = f.read().splitlines()

# Genel modeli çağırıyoruz (is_transfer=False)
model = get_model(is_transfer=False)

def predict(img_path):
    print(f"Analiz ediliyor: {img_path}")
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    
    result = model.predict(img_array[np.newaxis, ...])
    predicted_index = np.argmax(result)
    
    print("\n" + "="*30)
    print(f"TAHMİN: {labels[predicted_index]}")
    print("="*30)

if __name__ == "__main__":
    predict("daisy.jpg")