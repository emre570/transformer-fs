# Transformer Mimarisi: Temel Yapı Taşları

Selam! Bu yazıda sana Transformer mimarisinin yapı taşlarını anlatacağım. Ama öyle akademik, kuru kuru değil. Beraber çay-kahve içerken karşılıklı konuşuyormuşuz gibi düşünebilirsin. Hedefimiz şu: "Bu nedir abi ya?" dediğin hiçbir yer kalmayacak. Her şeyi sade, örnekli ve gerektiği yerde biraz matematikle açıklayacağız.

Bu yazı Umar Jamil'in [Coding a Transformer from scratch on PyTorch](tab:https://www.youtube.com/watch?v=ISNdQcPhsts) videosundan bakılarak yazılmıştır.

Hazırsan başlıyoruz.

---

## Input Embeddings

Model kelimeyi anlayamaz. Ona "köpek", "merhaba" veya "GPT" yazdığında hiçbir şey ifade etmez. Bu yüzden ilk işimiz kelimeleri sayısal vektörlere çevirmek. Bu vektörler ne kadar büyükse (örneğin 512 boyutunda), kelimenin anlamını o kadar detaylı taşıyabilir.

Torch’ta bunu şöyle yapıyoruz:

```python
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

Ölçekleme (√d_model) neden var? Çünkü küçük sayılarla çalışmak modelin öğrenmesini zorlaştırıyor. Bu çarpan sayesinde değerler normalize oluyor, eğitim daha stabil ilerliyor.

---

## Positional Encoding 

Kelimeyi sayıya çevirdik ama, model kelimenin **nerede** olduğunu hâlâ bilmiyor. “Geldim eve” ile “Eve geldim” aynı embedding’leri üretir. O yüzden pozisyon bilgisi eklememiz lazım.

Nasıl yapıyoruz?
Her pozisyon için özel bir vektör hesaplıyoruz. Bu vektörleri kelime embedding’ine ekliyoruz. Matematiksel formüllerle şöyle:

\[PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\]
\[PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\]

Sinüs - kosinüs ne alaka? Çünkü bu fonksiyonlar periyodik. Model mesela 5. kelime ile 10. kelime arasındaki uzaklığı daha rahat öğrenebiliyor. Hem sin hem cos koymamız da, yön bilgisi gibi farklı açılardan pozisyonu temsil etmek için.

Küçük örnek:

- Cümle: "Ben eve gidiyorum"
- Positional Encoding vektörleri sin/cos sayesinde pozisyona göre farklı değerler taşıyacak.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
``` 

---

## Multi-Head Attention

Geldik olayın kalbine. Attention, modelin "bu kelime şu kelimeyle alakalı mı?" sorusunu sormasını sağlar. Multi-Head Attention ise bu soruyu **farklı açılardan birden fazla kez** sormasını sağlar.

Üç ana yapı var:
- **Query (Q):** Şu anda bakılan kelime (ben neye dikkat etmeliyim?)
- **Key (K):** Diğer kelimelerin kimliği (bakılacak yerler)
- **Value (V):** O kelimelerden alınacak bilgi

Formül:
\[Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V\]

Örnek:
> Cümle: "Ayşe topu Ali'ye attı çünkü _ yorgundu."
>
> "_" yerine ne gelecek? "Ayşe" mi "Ali" mi?
>
> Model Query olarak “çünkü” kelimesine bakıyor, Key olarak diğer kelimelere. Dikkat skorlarını hesaplıyor. En çok kime benziyorsa, oradan bilgi çekiyor (Value).

Head’leri neden çoğaltıyoruz?
Her head farklı ilişkiyi öğreniyor. Bir head gramer ilişkisine odaklanırken diğeri duygusal ilişkiye, bir diğeri zamanlamaya odaklanabilir.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.h = h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        def transform(x, linear):
            x = linear(x)
            x = x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            return x

        q = transform(q, self.w_q)
        k = transform(k, self.w_k)
        v = transform(v, self.w_v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)
```

---

## Feed Forward Network

Kelimeler arası ilişkiyi öğrendik. Şimdi sıra her kelimenin kendi içinde daha karmaşık özellikler çıkarmasında.

Bu aşamada tüm pozisyonlara aynı MLP (iki katmanlı tam bağlı ağ) uygulanır:

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

- İlk katman genişletir (d_model → d_ff)
- ReLU ile aktivasyon uygular
- İkinci katman geri düşürür (d_ff → d_model)

Yani kelimeyi daha detaylı analiz eder, sonra tekrar orijinal boyuta döndürür. Basit ama güçlü.

---

## Layer Normalization – Sakin Ol Champ...

Katmanlar arası aktivasyonlar bazen kontrolden çıkabilir. Çok büyük – çok küçük değerler öğrenmeyi zorlaştırır. İşte bu yüzden LayerNorm kullanılır.

Her örnek **kendi embedding boyutu** üzerinden normalize edilir:

\[\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\]

Burada:
- `x`: Giriş vektörü
- `μ`: Ortalama (mean)
- `σ²`: Varyans
- `ε`: Sayısal kararlılık için çok küçük bir sayı (genelde 1e-5 ya da 1e-6)
- `γ`, `β`: Öğrenilebilir parametreler (girişin dağılımını yeniden ölçekleyip kaydırmak için)

- Ortalamayı sıfır yapar, varyansı 1’e çeker
- Gamma ve beta ile model isterse bu dağılımı kaydırabilir veya ölçekleyebilir

Not: BatchNorm yerine LayerNorm kullanmamızın sebebi şu: BatchNorm tüm batch’in istatistiklerine bakar. LayerNorm ise örneği kendi içinde işler. Sequence modeller için LayerNorm çok daha uygundur.

```python
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

---

## Residual Connection

Derin modellerde giriş bilgisi katmanlar arasında kaybolabilir. Residual connection sayesinde model, giriş bilgilerini doğrudan çıktıya ekleyerek bu sorunu çözer.

```python
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

Bu hem gradient akışını kolaylaştırır hem de modelin daha derin katmanlara ulaşmasını sağlar. Transformer’daki her blokta bu bağlantı var.

---