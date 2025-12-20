import tensorflow as tf
import tensorflow_hub as hub

def get_model(num_classes=1001, is_transfer=False):
    if is_transfer:
        # 1. DURUM: Çiçek Eğitimi İçin (Transfer Learning)
        # Sadece özellik çıkarıcı katmanı alıyoruz (Son katman yok)
        feature_extractor_url = "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3"
        layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)
        # Üzerine sadece 5 çiçek sınıfı için yeni bir katman ekliyoruz
        model = tf.keras.Sequential([layer, tf.keras.layers.Dense(num_classes)])
    else:
        # 2. DURUM: Genel Test İçin (Goldfish veya Daisy testi)
        # 1000 nesneyi tanıyan tam hazır modeli alıyoruz
        classifier_url = "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1"
        model = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=(224, 224, 3))])
    
    model.build([None, 224, 224, 3])
    return model