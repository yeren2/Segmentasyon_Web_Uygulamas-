# 🚀 YOLO26 Yapay Zeka Tabanlı Nesne Segmentasyon Paneli

<p align="center">
  <img src="https://shields.io" alt="YOLO26">
  <img src="https://shields.io" alt="Flask">
  <img src="https://shields.io" alt="Frontend">
</p>

Bu proje, Ultralytics'in en güncel **YOLO26 SOTA segmentasyon modelini** kullanan, web arayüzü üzerinden yüklenen görsellerdeki nesneleri anlık olarak tespit edip sınır hatlarını (konturlarını) yarı şeffaf maskelerle çizen modern bir web uygulamasıdır.

---

## ✨ Özellikler

- 🌌 **Siberpunk / Dark Mode Arayüz:** Gözü yormayan, Inter fontuyla güçlendirilmiş tamamen modern ve dinamik frontend tasarımı.
- 🎛️ **YOLO26 Entegrasyonu:** En güncel nesne tespit ve segmentasyon mimarisiyle yüksek doğruluk oranı.
- ⚡ **Base64 Bellek Yönetimi:** Yüklenen görseller sunucuda kalıcı olarak saklanmaz; RAM üzerinde işlenerek doğrudan tarayıcıya güvenli Base64 formatında iletilir.
- 📊 **Detaylı Nesne Listesi:** Algılanan tüm nesneler, güven skorları (Confidence % Score) ile birlikte etiketler halinde anlık listelenir.

---

## 🛠️ Kullanılan Teknolojiler

- **Backend:** Python, Flask, OpenCV, NumPy
- **Yapay Zeka:** Ultralytics YOLO26 (`yolo26n-seg.pt`)
- **Frontend:** HTML5, CSS3 (Grid & Flexbox), Modern JavaScript (Fetch API & FormData)

---

## 🚀 Kurulum ve Çalıştırma

Projenizi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

### 1. Depoyu Klonlayın veya İndirin
```bash
git clone https://github.com
cd Segmentasyon_Web_Uyg
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install flask ultralytics opencv-python numpy
```

### 3. Uygulamayı Başlatın
```bash
python main.py
```

Uygulama başladıktan sonra tarayıcınızdan **`http://localhost:5000`** adresine giderek paneli kullanmaya başlayabilirsiniz! 🌟

---

## 📁 Proje Yapısı

```text
Segmentasyon_Web_Uyg/
│
├── templates/
│   └── index.html      # Siberpunk temalı web arayüzü
│
├── .gitignore          # Gereksiz/büyük dosyaları engelleme listesi
├── main.py             # Flask backend ve YOLO26 motoru
└── README.md           # Proje tanıtım belgesi
```

---

## 📝 Lisans

Bu proje eğitim ve geliştirme amaçlı açık kaynak olarak paylaşılmıştır. İstediğiniz gibi kurcalayabilir, bozabilir ve geliştirebilirsiniz! 🚀
