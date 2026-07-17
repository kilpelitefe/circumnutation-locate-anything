# Circumnutation Projesi — Sonuçlar Özeti

Bitki hareketlerini **locate-anything** (NVIDIA LocateAnything-3B, C++/ggml portu) ile
noktasal takip ederek inceleme.

**İki veri seti üzerinde çalışıldı:**
- **A) Lima Bean (ANA SONUÇ)** — Wikimedia Commons, renkli, yandan → locate-anything ÇALIŞIYOR ✅
- **B) Circumnutation Tracker örneği** — gri, tepeden → locate-anything YETERSİZ ❌ (değerlendirme bulgusu)

---

# BÖLÜM A — locate-anything ile fasulye fidesi hareket takibi (ANA SONUÇ)

**Veri:** Wikimedia Commons "Lima Bean Time Lapse", **David Marvin, CC BY 3.0**
(`data_wiki/lima_bean.webm`, 3567 kare, 1280×720, renkli, yandan, 3 fasulye fidesi).
Atıf zorunlu: David Marvin, CC BY 3.0, via Wikimedia Commons.

**Kullanılabilir aralık:** yeşil-maske taramasıyla belirlendi (`bean_scan.py`, 27 sn):
frame **2000-3324** = olgun toprak-üstü fide fazı (öncesi toprak-altı çimlenme, sonrası logo kartı).

**Pipeline:**
1. `bean_detect.py` — her 15. karede locate-anything tam-kare tespit, prompt: `seedling`, mode fast (~80 sn/kare, 89 kare ≈ 2 saat)
2. `bean_track.py` — kutuları sabit x-bantlarıyla 3 bitkiye ata → kutu merkezi izle →
   hareketli-medyan aykırı-değer reddi → interpolasyon → drift çıkar → salınım

**locate-anything performansı (BU veride mükemmel) — yoğun çalışma, 265 kare:**
| Ölçüt | Değer |
|---|---|
| Tam 3 kutu bulunan kare | **155/169** yeni kare (ort ~2.9 kutu/kare) |
| Bitki bazında atama kapsamı | **bitki1 %99, bitki2 %92, bitki3 %99** |
| Kimlik kararlılığı | Üç bitki ayrı x-bantlarında (~293/573/959), karışma yok |
| Reddedilen aykırı kare | 6 / 7 / 1 |

**Ölçülen hareket (kutu merkezi, drift çıkarılmış):**
| Bitki | Salınım std-x | Tepe-tepe-x | Toplam yol |
|---|---|---|---|
| 1 | **18.1 px** | 98.2 px | 1217 px |
| 2 | 10.1 px | 71.8 px | 882 px |
| 3 | 5.6 px | 29.2 px | 649 px |

Bitki 1 en hareketli, bitki 3 en durgun (~3× fark). Seyrek (89 kare) ve yoğun (265 kare)
çalışmalar **aynı sıralamayı ve benzer genlikleri** verdi (16.7/8.8/5.4 → 18.1/10.1/5.6)
→ genlik ölçümü sağlam ve örneklemeden bağımsız.

Bitki 1 en hareketli, bitki 3 en durgun (~3× fark). Bitki 1 frame 2700-2900 aralığında
belirgin bir salınım yapıyor (x: 295→440→235).

**Görseller:** `test_out/bean_final_tracks.png` ⭐ (trajektoriler + salınım),
`test_out/bean_detect.png` (tespit örneği), `test_out/bean_track_montage.png` (kare kare doğrulama),
`test_out/bean_scan.png` (aralık belirleme).

## Periyot analizi (`bean_period.py`) — YOĞUN ÖRNEKLEME SONUCU

Yöntem: doğrusal detrend (büyüme kayması çıkarıldı) → otokorelasyon + FFT.
**Örnekleme: her 5. kare = 265 örnek** (~3.8 sa locate-anything hesabı).
Zaman ölçeği YAKLAŞIK: Commons açıklaması "6 günde >1600 foto"; bitki görüntüsü ~3268 kare
→ 1 kare ≈ 2.6 dk → **13 dk/örnek**, pencere ≈ 58 sa.
**Çözülebilir aralık: 0.4 sa (Nyquist) .. 29 sa** → 1-3 sa'lik circumnutation'ı görebilecek güçte.

| Bitki | Otokorelasyon (x) | FFT | Yorum |
|---|---|---|---|
| 1 | 26.9 sa | 19.5 sa | sirkadiyen ölçek |
| 2 | 51.6 sa (x) / 24.9 sa (y) | 58.4 / 29.2 sa | x = seri boyu (artefakt), y sirkadiyen |
| 3 | tepe yok | 29.2 sa | zayıf |

### ⭐ ANA BULGU (negatif sonuç, SAĞLAM)
**1-3 saatlik circumnutation TESPİT EDİLMEDİ** — üstelik örnekleme bunu görecek güçteydi
(13 dk, Nyquist 26 dk). Hipotez testi: "kutu merkezi ucun salınımını sönümlüyor olabilir" →
**apex (kutu üst-orta) ayrı analiz edildi, AYNI periyotları verdi** (27.1 vs 26.9 sa;
FFT ikisinde de 19.5 sa) → hipotez çürütüldü. Hızlı circumnutation gerçekten yok.

### İkincil bulgu (kaba, zayıf kanıt)
Tüm bitkilerde ~20-29 sa'lik yavaş salınım var = **sirkadiyen ölçek** (video 6 gün/gece
döngüsü içeriyor → günlük yaprak hareketi makul). AMA:
- Pencerede sadece ~2 döngü var.
- FFT'nin "19.5 sa" ve "29.2 sa" değerleri **komşu frekans kutucukları** (265/3 ve 265/2 örnek)
  → bu ölçekte periyot çözünürlüğü kaba, ikisi ayırt edilemez.
- Bitki 2'nin x'indeki 51-58 sa = serinin kendisi = artefakt, döngü değil.

**Dürüst özet: "yavaş, sirkadiyen ölçekte bir salınım var" denebilir; kesin periyot verilemez.**

**Neden circumnutation yok?** Kesin bilinmiyor. Olasılıklar: fideler aktif circumnutation
fazını geçmiş olabilir; video düzenlenmiş/farklı sahneler birleştirilmiş olabilir (sürekli
zaman ekseni bozulur); 2B yandan görüntüde 3B dairesel hareketin izdüşümü zayıf kalabilir.

**Görsel:** `test_out/bean_period.png` (detrend sinyaller + otokorelasyon)

**Genel sınırlama:** Bu bir *büyüme* time-lapse'i; sinyal büyüme + nutasyon + yaprak hareketi
karışımı. Kontrollü bir circumnutation deneyi değil, o yüzden temiz dairesel döngüler
(Bölüm B'deki gibi) beklenmemeli.

---

# BÖLÜM B — Circumnutation Tracker örnek verisi (locate-anything DEĞERLENDİRMESİ)

Video: `Video 1.avi` (757 kare, 768×576, gri, tepeden, 16 fide; her kare 5 dk arayla).
Üst 8 fide = distilled water, alt 8 fide = nutrient solution. Aynı video.

---

## 1. locate-anything (VLM) DEĞERLENDİRMESİ

Zorunlu kullanım için test edildi. **Bu veride güvenilmez** çıktı.

| Deneme | Sonuç |
|---|---|
| Bitki-başına küçük crop (büyütme/CLAHE/kontrast dahil) | **0 tespit** — model izole küçük bitkiyi crop'ta hiç bulamıyor |
| Tam-kare, çıplak "small plant" prompt | Kapsam düşük/tutarsız (kare başına 0–17 kutu) |
| Tam-kare, şablon "Locate all the instances...: seedling." | Daha iyi (frame 500: 1→15) ama bazı kareler yine 0 |
| Tam-kare "plant"/"leaf"/"sprout" | 26 kutu (üst sınır) ama videoya gömülü YAZILARI da yakalıyor |

**Doğrulama (6 kare, DB gerçek referansına karşı):**
- Kapsam: distilled %27, nutrient %19 (yani plant-kare çiftlerinin ~3/4'ünde tespit YOK)
- Hata (bulduğunda): 18–27 px
- Hız: ~80 sn/kare (CPU) → tüm video ~17 saat
- Kutular sık sık 2 fideyi birleştiriyor; kutu merkezi büyüyen uçtan kayık

Görsel kanıt: `test_out/la_vs_truth.png` (kırmızı=LA kutu, sarı=merkez, yeşil=gerçek uç)
Ham tespitler: `test_out/la_boxes/*.json`, kareler: `test_out/la_frames/*.png`

---

## 2. Klasik CV takibi (karşılaştırma / referans yöntem)

Origin'den crop → Otsu eşikleme → en büyük koyu kontur centroid'i.
- Hata: **ort 9.3 px, medyan 8.8 px** (DB insan-takibine karşı)
- Hız: **tüm video ~10 sn** (locate-anything'ten ~binlerce kat hızlı)
- Görsel: `test_out/validation_all8.png`

---

## 3. BİLİMSEL BULGU: tedavi karşılaştırması

Circumnutation metrikleri (çimlenme sonrası, drift çıkarılmış salınım):

| Metrik | Saf su (CV) | Saf su (insan/DB) | Besin (CV) | Besin (insan/DB) |
|---|---|---|---|---|
| Periyot | 7.1 sa | 6.2 sa | **3.9 sa** | **3.9 sa** |
| Genlik | 3.9 px | 3.1 px | **15.0 px** | **21.5 px** |
| Net dönme | 3.0 tur | 3.7 tur | **8.3 tur** | **9.7 tur** |
| Yol uzunluğu | 377 px | 314 px | **1589 px** | **1996 px** |

**Sonuç:** Besin çözeltisi circumnutation'ı belirgin şekilde hızlandırıyor (periyot ~6-7→4 sa)
ve genlik/aktiviteyi ~5-7 kat artırıyor. Hem otomatik CV hem insan takibi AYNI yönde →
gerçek biyolojik etki, izleme artefaktı değil.

Görseller: `test_out/treatment_compare.png`, `test_out/circumnutation_detrended.png`

---

## Dosyalar

**Kod (repo kökü):**
- `la_detect.py` — locate-anything tam-kare tespit (ADIM 1)
- `la_analyze.py` — kutu→fide atama + DB doğrulama (ADIM 2)
- `ct_track.py` / `ct_analyze.py` / `ct_compare.py` — klasik CV takip + analiz

**Sonuç görselleri (`test_out/`):**
- `la_vs_truth.png` — locate-anything kutuları vs gerçek uçlar ⭐ (VLM değerlendirmesi)
- `treatment_compare.png` — saf su vs besin karşılaştırması ⭐ (bilimsel bulgu)
- `validation_all8.png` — CV takip doğrulaması
- `circumnutation_detrended.png` — salınım döngüleri

## Sınırlamalar
- n=8/grup, henüz formal istatistik testi yok (etki büyük).
- İlk ~150 kare (çimlenme öncesi) analizden çıkarıldı.
- Besin grubu izlemesi biraz daha gürültülü (14.5 px) ama etki bunu kat kat aşıyor.
