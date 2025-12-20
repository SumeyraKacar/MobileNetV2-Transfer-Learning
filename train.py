from model import get_model

def show_transfer_learning():
    print("Transfer Learning Mimarisi Oluşturuluyor...")
    # Çiçekler için (is_transfer=True) ve 5 sınıf (papatya, gül vb.)
    model = get_model(num_classes=5, is_transfer=True)
    model.summary()
    print("\nModelin son katmanı 5 sınıf için özelleştirildi.")

if __name__ == "__main__":
    show_transfer_learning()