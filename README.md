# 📊 CNN ile MNIST Sayı Tanıma Projesi

Bu proje, TensorFlow ve Keras kullanarak el yazısı rakamları tanıyan bir Yapay Sinir Ağı (Neural Network) uygulamasıdır.

## 🎯 Proje Amacı

MNIST veri setindeki el yazısı rakamları (0-9) otomatik olarak tanıyan bir derin öğrenme modeli geliştirmek.

## 📚 Veri Seti Hakkında

MNIST veri seti:
- 60,000 eğitim görüntüsü
- 10,000 test görüntüsü
- 28x28 piksel siyah-beyaz görüntüler
- 0-9 arası rakamların el yazısı örnekleri

## 🔧 Model Mimarisi

### İlk Model (Basit)
```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
```

### Geliştirilmiş Model
```python
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
```

## 🛠️ Veri Ön İşleme Adımları

1. **Veri Yükleme**
```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

2. **Normalizasyon**
```python
X_train = X_train / 255
X_test = X_test / 255
```

3. **Veri Şekillendirme**
```python
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
```

## 📈 Model Eğitimi ve Değerlendirme

### Model Konfigürasyonu
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Eğitim Parametreleri
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 5
- Batch Size: Default (32)

## 📊 Performans Analizi

1. **Confusion Matrix**
- Modelin tahmin performansını detaylı analiz
- Her sınıf için doğru ve yanlış tahminlerin görselleştirilmesi
- Seaborn heatmap ile görselleştirme

2. **Tahmin Örnekleri**
```python
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
```

## 💡 Model İyileştirmeleri

1. **Mimari İyileştirmeler**
   - Gizli katman ekleme (100 nöron)
   - ReLU aktivasyon fonksiyonu kullanımı
   - Dropout katmanı ekleme potansiyeli

2. **Hiperparametre Optimizasyonu**
   - Learning rate ayarlaması
   - Batch size optimizasyonu
   - Epoch sayısı ayarlaması

## 📉 Görselleştirme Araçları

1. **Matplotlib**
```python
plt.matshow(X_test[264])  # Örnek görüntü gösterimi
```

2. **Seaborn**
```python
sn.heatmap(cm, annot=True, fmt='d')  # Confusion matrix görselleştirme
```

## 🔍 Model Değerlendirme Metrikleri

1. **Accuracy**
   - Eğitim ve test verisi üzerinde doğruluk oranı

2. **Confusion Matrix**
   - Sınıf bazında başarı analizi
   - Yanlış sınıflandırmaların tespiti

## 🚀 İleri Seviye İyileştirmeler

1. **Model Mimarisi**
   - Convolutional katmanlar ekleme
   - Batch normalization
   - Max pooling katmanları

2. **Veri Artırma (Data Augmentation)**
   - Görüntü döndürme
   - Ölçeklendirme
   - Gürültü ekleme

3. **Regularizasyon**
   - Dropout
   - L1/L2 regularization
   - Early stopping

## 📌 Önemli Notlar

1. Veri normalizasyonu model performansı için kritik
2. Model mimarisinde gizli katman eklenmesi başarıyı artırır
3. Confusion matrix analizi hata kaynaklarını belirlemeye yardımcı olur
4. ReLU aktivasyonu sigmoid'e göre daha iyi performans gösterir

## 🔮 Gelecek Geliştirmeler

1. CNN (Convolutional Neural Network) implementasyonu
2. Daha derin model mimarileri deneme
3. Transfer learning uygulama
4. Real-time tahmin sistemi geliştirme
