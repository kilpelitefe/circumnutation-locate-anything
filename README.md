# Bitki Hareketi Takibi — locate-anything ile Noktasal Konumlandırma

> ## 🚧 Durum: DEVAM EDİYOR
> Bu proje aktif geliştirme aşamasındadır. Çalışan bir pipeline ve doğrulanmış sonuçlar
> mevcut; ancak asıl hedef (locate-anything ile **circumnutation** yakalamak) henüz tam
> tutturulamadı. Güncel adım: daha uygun bir time-lapse veri seti seçimi.
> Ayrıntı: [Yol haritası](#yol-haritası--devam-eden-işler)

Açık-kelime dağarcıklı bir görsel-dil modeli olan
[**locate-anything**](https://github.com/mudler/locate-anything.cpp)
(NVIDIA LocateAnything-3B'nin C++/ggml portu) kullanılarak, bitkilerin zaman-atlamalı
(time-lapse) videolarındaki hareketlerinin **noktasal konumlandırma** ile incelenmesi.

Klasik yaklaşımda araştırmacı her karede bitkinin ucuna elle tıklar. Buradaki amaç, bunu
bir metin prompt'u (`"seedling"`) ile otomatikleştirmek.

---

## Neden ilginç: aracın sınırlarını bulmak

Projenin çıktısı sadece "çalışan bir takip aracı" değil; **locate-anything'in hangi
görüntü tipinde çalışıp hangisinde çalışmadığının ölçülmüş haritası.** İki zıt veri
setinde denendi ve sonuç dramatik biçimde farklı çıktı:

| Veri seti | Görüntü tipi | Kapsam | Sonuç |
|---|---|---|---|
| Lima Bean (Wikimedia) | Renkli, yandan, koyu zemin | **%92–99** | ✅ Çalışıyor |
| Circumnutation Tracker | Gri, tepeden, düşük kontrast | **%19–27** | ❌ Yetersiz |

Model doğal görüntülerde (COCO tarzı) eğitildiği için, ikinci veri setinde bitkileri değil
**videoya gömülü yazıları** tespit etti.

---

## Nasıl ilerledik (metodoloji)

Proje doğrusal ilerlemedi; her adım bir öncekinin bulgusuyla şekillendi. Bu bölüm o
mantığı kaydeder — çünkü asıl öğretici olan kısım burası.

### 1. Başlangıç: locate-anything'i circumnutation verisine uygula
Elde [Circumnutation Tracker](https://plantmethods.biomedcentral.com/articles/10.1186/1746-4811-10-24)
örnek verisi vardı: 757 kare, gri, tepeden, 16 fide, elle işaretlenmiş **gerçek referans**
(SQLite). Plan: her bitkiyi kırp → locate-anything ile ucu bul → izle.

### 2. Bulgu: model bu veride çalışmıyor
- Bitki-başına crop → **0 tespit** (büyütme, CLAHE, kontrast germe denendi; hiçbiri işe yaramadı)
- Tam-kare, `plant`/`leaf` → **yazıları** tespit ediyor ("1", "5", "distiled water", zaman damgası)
- Tam-kare, `small plant` → bitkileri buluyor ama kapsam %19–27, kutular fideleri birleştiriyor
- Hız: ~80 sn/kare → tüm video ~17 saat

**Karar:** Bu bir başarısızlık değil, **ölçülmüş bir bulgu.** Belgelendi (bkz.
`figures/la_vs_truth.png`).

### 3. Referans nokta: klasik görüntü işleme
locate-anything'in performansını bağlama oturtmak için basit bir yöntem kuruldu:
Otsu eşikleme → en büyük koyu kontur → ağırlık merkezi.

| Yöntem | Kapsam | Hata | Süre |
|---|---|---|---|
| locate-anything | %19–27 | 18–27 px | ~17 saat |
| Klasik CV | %100 | **9.3 px** | **10 saniye** |

Bu, kontrollü/gri veride VLM'in neden yanlış araç olduğunu sayısallaştırdı.

### 4. Yan ürün: gerçek bir biyolojik bulgu
Klasik CV yeterince güvenilir olduğu için deneyin asıl sorusu cevaplanabildi. Aynı videoda
üstteki 8 fide saf suda, alttaki 8 fide besin çözeltisinde:

> **Besin çözeltisi circumnutation'ı ~2× hızlandırıyor** (periyot ~6–7 sa → ~4 sa)
> **ve genlik/aktiviteyi 5–7× artırıyor.**
> Hem otomatik hem elle takipte aynı yönde → izleme artefaktı değil.

### 5. Pivot: aracın güçlü olduğu veriyi seç
locate-anything'i veriye zorlamak yerine, ona uygun veri seçildi: **renkli, yandan, net
bitki.** Wikimedia Commons'tan
[Lima Bean Time Lapse](https://commons.wikimedia.org/wiki/File:Lima_Bean_Time_Lapse.webm)
(CC BY 3.0) — akademik kullanım için açık lisanslı ve atıf verilebilir.

Sonuç: model **tam 3 fideyi** temiz kutularla buldu, kimlik kararlı kaldı.
(`figures/bean_detect.png`)

### 6. Pipeline: tespit → atama → izleme → analiz
```
bean_scan.py     Ucuz yeşil-maske taraması (27 sn) → kullanılabilir kare aralığını bul
                 (intro toprak-altı çimlenme ve outro logo kartını dışla)
     ↓
bean_detect.py   locate-anything tam-kare tespiti, prompt "seedling" (~80 sn/kare)
                 → data/det_boxes/*.json  [pahalı adım; çıktılar repoda kayıtlı]
     ↓
bean_track.py    Kutuları sabit x-bantlarıyla bitkilere ata → kutu merkezini izle
                 → aykırı değer reddi → interpolasyon → data/bean_tracks.npz
     ↓
bean_period.py   Doğrusal detrend → otokorelasyon + FFT ile periyot
```
Ucuz bir ön-tarama ile pahalı VLM'i doğru yere harcamak, tekrarlanan bir tasarım kararı oldu.

### 7. Bulgu: circumnutation yok (sağlam negatif sonuç)
Örnekleme 6× yoğunlaştırıldı (13 dk/örnek, Nyquist 26 dk) — **1–3 saatlik circumnutation'ı
görecek güçteydik ve yoktu.** En olası itiraz da test edilip çürütüldü: *"kutu merkezi ucun
salınımını sönümlüyor olabilir"* → **apex ayrı analiz edildi, birebir aynı periyotları verdi**
(27.1 vs 26.9 sa).

Bu "ölçemedik" değil, **"yeterli çözünürlükle baktık ve yoktu"** demektir.

Sebep muhtemelen veri seçimi: Lima Bean bir **büyüme** time-lapse'i (~2.6 dk/kare), kontrollü
bir circumnutation deneyi değil.

### 8. Şu an: daha iyi veri seti seçimi
Ortaya şu ironi çıktı:
- locate-anything'in **çalıştığı** veri (fasulye) → circumnutation **göstermiyor**
- circumnutation **gösteren** veri (CT) → locate-anything **çalışmıyor**

Aranan: **her ikisini birden** sağlayan video. Commons'ın
[*Time-lapse videos of plants*](https://commons.wikimedia.org/wiki/Category:Time-lapse_videos_of_plants)
kategorisi (71 video) tarandı. Öne çıkan aday:
[**Pea de-étiolation with circadian cycle**](https://commons.wikimedia.org/wiki/File:Pea_de-%C3%A9tiolation_with_circadian_cycle.ogv)
(CC BY-SA 3.0) — temiz koyu zemin (locate-anything'in sevdiği kurulum) **+ 51 saat, ~40 sn/kare**
(periyot başına 90–180 kare) **+ 16sa ışık / 8sa karanlık** sirkadiyen döngü.

---

## Sonuçlar

Tam rapor: **[SONUCLAR.md](SONUCLAR.md)** · Görsel rapor: **[rapor.html](rapor.html)**

### locate-anything performansı (fasulye, 265 kare)
| Ölçüt | Değer |
|---|---|
| Tam 3 kutu bulunan kare | 155 / 169 |
| Kapsam (bitki 1 / 2 / 3) | %99 / %92 / %99 |
| Kimlik kararlılığı | Ayrı x-bantlarında, karışma yok |

### Ölçülen hareket
| Bitki | Salınım std-x | Tepe-tepe | Toplam yol |
|---|---|---|---|
| 1 | **18.1 px** | 98.2 px | 1217 px |
| 2 | 10.1 px | 71.8 px | 882 px |
| 3 | 5.6 px | 29.2 px | 649 px |

Seyrek (89 kare) ve yoğun (265 kare) çalışmalar aynı sıralamayı verdi → genlik ölçümü
örneklemeden bağımsız, sağlam.

![Takip sonuçları](figures/bean_final_tracks.png)

---

## Depo yapısı

```
├── bean_scan.py / bean_detect.py / bean_track.py / bean_period.py
│                          Ana pipeline (Lima Bean, locate-anything)
├── la_detect.py / la_analyze.py / test_one_crop.py
│                          locate-anything değerlendirmesi (gri veri)
├── ct_track.py / ct_analyze.py / ct_compare.py
│                          Klasik CV referans yöntemi + tedavi karşılaştırması
├── data/
│   ├── det_boxes/*.json   locate-anything tespitleri, fasulye (265 kare ≈ 3.8 saat hesap)
│   ├── la_boxes/*.json    locate-anything tespitleri, gri veri
│   └── bean_tracks.npz    işlenmiş izler
├── figures/               sonuç görselleri
├── SONUCLAR.md            ayrıntılı rapor
└── rapor.html             görsel rapor (tek dosya, görseller gömülü)
```

**Not:** `data/det_boxes/` kayıtlı olduğu için, analiz adımlarını (`bean_track.py`,
`bean_period.py`) **3.8 saatlik tespiti tekrar çalıştırmadan** yeniden üretebilirsiniz.

---

## Kurulum

Bu depo **analiz kodunu** içerir; motor ve model ayrıca gerekir.

**1. locate-anything.cpp derle** (CPU yeterli):
```bash
git clone --recursive https://github.com/mudler/locate-anything.cpp
cd locate-anything.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64 -DLA_BUILD_CLI=ON
cmake --build build --config Release -j
```

**2. Model indir** (~6.3 GB, `q8_0` önerilen):
[mudler/locate-anything.cpp-gguf](https://huggingface.co/mudler/locate-anything.cpp-gguf)
→ `models/locate-anything-q8_0.gguf`

> **GPU notu:** q8_0 ~6.3 GB ağırlık + aktivasyon ister; 4 GB VRAM'e **sığmaz**.
> Bu proje CPU-only çalıştırıldı (i5-12450H, ~80 sn/kare). CUDA derlemesi gereksiz.

**3. Python bağımlılıkları:**
```bash
pip install opencv-python numpy matplotlib
```

**4. Video indir:** [Lima Bean Time Lapse](https://commons.wikimedia.org/wiki/File:Lima_Bean_Time_Lapse.webm)
→ `data_wiki/lima_bean.webm`

---

## Teknik notlar (tuzaklar)

Üçü de kodda yorumlandı; benzer kurulumlarda tekrar edebilir:

1. **Türkçe yol × CLI argümanı** — Proje yolu `Masaüstü` içerdiğinde, CLI'a *mutlak* yol
   verilince MSVC'nin ANSI `argv`'si "ü"yü bozuyor. **Çözüm:** çalıştırılabilire mutlak yol
   (CreateProcessW Unicode-güvenli), model/girdi/çıktıya **göreli ASCII yol** + `cwd=repo`.

2. **Türkçe yol × `cv2.imwrite`** — OpenCV mutlak ANSI yola *sessizce* bozuk dosya yazıyor;
   CLI bozuk kareyi okuyup "0 tespit" döndürüyor. İlk denemenin sıfır vermesi modelin değil
   bunun suçuydu. **Çözüm:** `cv2.imencode('.png', img)[1].tofile(path)`.

3. **Aykırı değer reddinde zincirleme** — Ardışık noktaya göre reddetme, bitki büyüme
   kaymasıyla uzaklaşınca çöküyor (87 karenin 86'sı reddedildi). **Çözüm:** yerel
   **hareketli medyandan sapma** (eşik 35 px, pencere 7) — yalnızca izole sıçramaları atar.

---

## Yol haritası / devam eden işler

- [ ] **Bezelye veri setine geçiş** — locate-anything'i `Pea de-étiolation` videosunda test
      et; circumnutation'ı yakalayacak zamansal çözünürlük burada var
- [ ] **Yol yapılandırması** — Scriptlerde sabit kodlanmış yollar var (CT verisi geçici bir
      dizini işaret ediyor). Bir config dosyasına taşınmalı; şu an başka makinede
      olduğu gibi çalışmaz
- [ ] **Gerçek zaman ölçeği** — Fasulye videosunun çekim aralığı bilinmiyor ("6 gün / 3268
      kare" varsayımı kullanıldı). Saat cinsinden periyotlar bu yüzden yaklaşık
- [ ] **İstatistik testi** — Gri veri tedavi karşılaştırmasında n=8/grup, formal test yok
- [ ] Kod tekrarının azaltılması (`ct_*` ve `bean_*` pipeline'ları benzer adımları paylaşıyor)

---

## Atıflar ve lisanslar

**Veri (atıf zorunlu):**
- *Lima Bean Time Lapse* — David Marvin, **CC BY 3.0**, via Wikimedia Commons
- *Pea de-étiolation with circadian cycle* — D. Bornand, G. Ciprietti, S. Zbinden ve ark.,
  **CC BY-SA 3.0**, via Wikimedia Commons *(planlanan)*
- *Circumnutation Tracker* örnek veri seti — Stolarz ve ark., Plant Methods 2014

**Yazılım:**
- [locate-anything.cpp](https://github.com/mudler/locate-anything.cpp) — Ettore Di Giacinto
  & Richard Palethorpe, MIT
- [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — NVIDIA
  (kendi model lisansı)
