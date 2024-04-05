Multi-Head Attention ve RMSNorm ile LSTM

1. Gömme Katmanı (Embeddings):
Kelimeleri yoğun vektör gösterimlerine dönüştürür.

2. LSTM Katmanı:
Giriş dizisindeki uzun vadeli bağımlılıkları yakalamak için kullanılır.

3. Dikkat Mekanizması (Multi-Head Attention):
Giriş dizisindeki farklı bölümlere odaklanmak ve bunlar arasındaki ilişkileri öğrenmek için kullanılır.
MultiHeadAttention sınıfı, çoklu dikkat başlıkları kullanarak daha zengin ve bağlamsal bir temsil oluşturur.

4. RMSNorm Katmanı:
Katman normalleştirme yöntemidir.
Geleneksel katman normalleştirmenin aksine, RMSNorm girdileri karelerinin ortalama kareköküne bölerek normalleştirir.

5. Feedforward Katmanı:
Dikkat katmanından sonra ek bilgi çıkarmak için kullanılır.
Burada örnek olarak 512 nöronlu bir gizli katman, ReLU aktivasyon fonksiyonu ve dropout kullanılmıştır.

6. Normalize Katmanı:
Feedforward katmanından sonra ek bir normalleştirme adımı eklenir.
Burada örnek olarak Layer Normalization kullanılmıştır.

7. Çıkış Katmanı (FC):
Sonuçları tahmin edilen kelime olasılıklarına dönüştürür.
Modelin Eğitimi:
Model, bir kelime sözlüğü ve ilgili parametrelerle oluşturulur.


Önemli Noktalar:
Model, hem uzun vadeli bağımlılıkları yakalamak için LSTM'yi hem de giriş dizisi içindeki ilişkileri modellemek için dikkat mekanizmasını kullanır.
RMSNorm, modelin daha kararlı bir şekilde eğitilmesine yardımcı olur.
Feedforward ve normalize katmanları, modelin ifade gücünü artırır.

![Ekran Görüntüsü (23)](https://github.com/hasnbayat/model-machine-ai/assets/165487438/416ec0cf-6b4a-4163-9e24-0a64de7a5874)
