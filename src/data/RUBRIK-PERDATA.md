# Rubrik Analisis Unsur Hukum Perdata (KUHPerdata)

Dokumen ini merupakan adaptasi dari **rubrik analisis unsur tindak pidana** yang diberikan oleh Law Supervisor untuk domain hukum pidana. Rubrik ini diterjemahkan ke domain **hukum perdata** (KUHPerdata) dengan tujuan:

1. Memberikan kerangka sistematis untuk menilai relevansi pasal KUHPerdata terhadap fakta kasus
2. Digunakan sebagai template prompt LLM untuk ekspansi ground truth dataset
3. Mengurangi hub bias dengan mengidentifikasi pasal-pasal yang relevan namun tidak dikutip secara eksplisit

## Struktur Rubrik

Sama seperti rubrik pidana, setiap kasus dianalisis dengan:

| Komponen | Rubrik Pidana | Rubrik Perdata |
|----------|---------------|----------------|
| **Pasal yang dianalisis** | UU Tindak Pidana Korupsi | Pasal KUHPerdata |
| **Unsur** | Unsur Tindak Pidana | Unsur Hukum Perdata |
| **Fakta** | Fakta Perbuatan yang dilakukan | Fakta Kasus yang Memenuhi Unsur |
| **Dasar** | Alat Bukti yang mendukung | Dasar Pertimbangan Hukum |
| **Kesimpulan** | Unsur terpenuhi → terbukti | Unsur terpenuhi → pasal relevan |

**Prinsip utama**: Sebuah pasal dinyatakan **RELEVAN** terhadap kasus jika dan hanya jika **seluruh unsur pokok** pasal tersebut dapat dipetakan ke fakta dalam kasus. Jika ada unsur pokok yang tidak terpenuhi oleh fakta, pasal dinyatakan **TIDAK RELEVAN**.

---

## Kategori Pasal KUHPerdata dan Unsur-Unsurnya

### A. Perbuatan Melawan Hukum (PMH) — Pasal 1365-1367

| Pasal | Unsur-Unsur |
|-------|-------------|
| **1365** | (1) adanya perbuatan, (2) perbuatan tersebut melanggar hukum, (3) adanya kerugian, (4) adanya kesalahan (kesengajaan atau kelalaian), (5) adanya hubungan kausal antara perbuatan dan kerugian |
| **1366** | (1) adanya kelalaian atau kesembronoan, (2) adanya kerugian yang ditimbulkan, (3) hubungan kausal antara kelalaian dan kerugian |
| **1367** | (1) adanya hubungan subordinasi/pengawasan (orangtua-anak, majikan-bawahan, guru-murid), (2) perbuatan dilakukan oleh pihak yang berada di bawah tanggung jawab, (3) perbuatan terjadi dalam konteks hubungan tersebut, (4) adanya kerugian |

### B. Perjanjian — Pasal 1313, 1320, 1338

| Pasal | Unsur-Unsur |
|-------|-------------|
| **1313** | (1) adanya perbuatan mengikatkan diri, (2) dilakukan oleh satu orang atau lebih terhadap pihak lain |
| **1320** | (1) kesepakatan para pihak, (2) kecakapan para pihak, (3) suatu pokok persoalan tertentu (objek perjanjian), (4) sebab/kausa yang tidak terlarang |
| **1338** | (1) adanya persetujuan yang dibuat sesuai undang-undang, (2) berlaku sebagai undang-undang bagi para pihak, (3) pelaksanaan dengan itikad baik |

### C. Wanprestasi / Ingkar Janji — Pasal 1234, 1238, 1243

| Pasal | Unsur-Unsur |
|-------|-------------|
| **1234** | (1) adanya perikatan, (2) perikatan berupa memberikan sesuatu / berbuat sesuatu / tidak berbuat sesuatu |
| **1238** | (1) adanya debitur, (2) adanya pernyataan lalai (somasi/surat perintah/akta/lewat waktu), (3) kewajiban yang belum dipenuhi |
| **1243** | (1) adanya perikatan yang tidak dipenuhi, (2) debitur telah dinyatakan lalai, (3) debitur tetap lalai, (4) adanya biaya/kerugian/bunga yang ditimbulkan |

### D. Hak Milik / Kebendaan — Pasal 570, 584

| Pasal | Unsur-Unsur |
|-------|-------------|
| **570** | (1) adanya barang yang dimiliki, (2) hak menikmati dan berbuat secara bebas, (3) tidak bertentangan dengan undang-undang/peraturan umum, (4) tidak mengganggu hak orang lain |
| **584** | (1) adanya barang yang menjadi objek hak milik, (2) cara perolehan: pengambilan / perlekatan / lewat waktu / pewarisan / penyerahan, (3) dilakukan oleh orang yang berhak |

### E. Pewarisan — Pasal 832, 833

| Pasal | Unsur-Unsur |
|-------|-------------|
| **832** | (1) adanya orang yang meninggal, (2) adanya keluarga sedarah yang sah atau di luar perkawinan, atau suami/istri yang masih hidup, (3) harta peninggalan yang menjadi objek waris |
| **833** | (1) adanya ahli waris, (2) peralihan hak milik secara otomatis karena hukum, (3) meliputi semua barang, hak, dan piutang pewaris |

### F. Kedewasaan / Kecakapan — Pasal 330

| Pasal | Unsur-Unsur |
|-------|-------------|
| **330** | (1) usia belum genap 21 tahun, (2) belum pernah kawin, (3) konsekuensi: berada di bawah kekuasaan orangtua atau perwalian |

---

## Contoh Kasus Pertama: Perjanjian Kredit dan Lelang Jaminan

### Deskripsi Kasus

> Penggugat dan Tergugat I memiliki hubungan utang piutang berdasarkan perjanjian kredit yang dijamin dengan dua bidang tanah. Penggugat tidak melunasi kewajiban pembayaran angsuran sesuai jadwal meskipun telah diberikan surat peringatan, sehingga dinyatakan melakukan ingkar janji. Tergugat I kemudian melaksanakan haknya untuk menjual objek jaminan melalui pelelangan umum karena Penggugat dianggap lalai memenuhi perikatannya. Penggugat mengajukan keberatan dengan dalil adanya keadaan memaksa akibat pandemi, namun Tergugat I menegaskan bahwa pelaksanaan lelang adalah konsekuensi hukum dari ingkar janji yang telah terjadi.

### Analisis Kasus Pertama

#### Analisis terhadap Pasal 1320 — Syarat Sah Perjanjian

> **Pasal 1320**: Supaya terjadi persetujuan yang sah, perlu dipenuhi empat syarat: 1. kesepakatan mereka yang mengikatkan dirinya; 2. kecakapan untuk membuat suatu perikatan; 3. suatu pokok persoalan tertentu; 4. suatu sebab yang tidak terlarang.

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Kesepakatan para pihak | Penggugat dan Tergugat I sepakat melakukan perjanjian kredit | Adanya perjanjian kredit yang mengikat kedua belah pihak |
| 2 | Kecakapan para pihak | Penggugat (debitur) dan Tergugat I (kreditur/bank) adalah pihak yang cakap hukum | Bank sebagai badan hukum dan Penggugat sebagai subjek hukum dewasa |
| 3 | Suatu pokok persoalan tertentu | Objek perjanjian berupa kredit dengan jaminan dua bidang tanah, dengan jadwal pembayaran angsuran | Objek perjanjian jelas dan tertentu |
| 4 | Sebab yang tidak terlarang | Perjanjian kredit dengan jaminan adalah perbuatan hukum yang sah | Tidak bertentangan dengan undang-undang, kesusilaan, atau ketertiban umum |

**KESIMPULAN**: Keempat unsur Pasal 1320 **terpenuhi**. Perjanjian kredit antara Penggugat dan Tergugat I adalah perjanjian yang sah menurut hukum. → **RELEVAN**

---

#### Analisis terhadap Pasal 1238 — Pernyataan Lalai

> **Pasal 1238**: Debitur dinyatakan lalai dengan surat perintah, atau dengan akta sejenis itu, atau berdasarkan kekuatan dari perikatan sendiri, yaitu bila perikatan ini mengakibatkan debitur harus dianggap lalai dengan lewatnya waktu yang ditentukan.

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Adanya debitur | Penggugat adalah debitur dalam perjanjian kredit | Hubungan hukum utang-piutang berdasarkan perjanjian kredit |
| 2 | Pernyataan lalai | Penggugat telah diberikan surat peringatan oleh Tergugat I | Surat peringatan merupakan bentuk somasi sebagaimana dimaksud pasal ini |
| 3 | Kewajiban yang belum dipenuhi | Penggugat tidak melunasi kewajiban pembayaran angsuran sesuai jadwal | Lewatnya waktu pembayaran yang ditentukan dalam perjanjian |

**KESIMPULAN**: Ketiga unsur Pasal 1238 **terpenuhi**. Penggugat telah dinyatakan lalai secara sah melalui surat peringatan. → **RELEVAN**

---

#### Analisis terhadap Pasal 1243 — Ganti Rugi atas Wanprestasi

> **Pasal 1243**: Penggantian biaya, kerugian dan bunga karena tak dipenuhinya suatu perikatan mulai diwajibkan, bila debitur, walaupun telah dinyatakan lalai, tetap lalai untuk memenuhi perikatan itu, atau jika sesuatu yang harus diberikan atau dilakukannya hanya dapat diberikan atau dilakukannya dalam waktu yang melampaui waktu yang telah ditentukan.

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Adanya perikatan yang tidak dipenuhi | Penggugat tidak memenuhi kewajiban pembayaran angsuran kredit | Ingkar janji terhadap perjanjian kredit |
| 2 | Debitur telah dinyatakan lalai | Tergugat I telah memberikan surat peringatan kepada Penggugat | Somasi telah dilakukan sesuai Pasal 1238 |
| 3 | Debitur tetap lalai | Meskipun telah diperingatkan, Penggugat tetap tidak melunasi kewajibannya | Penggugat berdalih keadaan memaksa (pandemi) tetapi tidak mengubah status wanprestasi |
| 4 | Adanya kerugian | Tergugat I mengalami kerugian akibat kredit macet, sehingga melaksanakan lelang jaminan | Pelaksanaan lelang sebagai konsekuensi hukum dari kerugian kreditur |

**KESIMPULAN**: Keempat unsur Pasal 1243 **terpenuhi**. Penggugat wajib menanggung biaya, kerugian, dan bunga akibat wanprestasi. → **RELEVAN**

---

#### Analisis terhadap Pasal 570 — Hak Milik (sebagai contoh TIDAK RELEVAN)

> **Pasal 570**: Hak milik adalah hak untuk menikmati suatu barang secara lebih leluasa dan untuk berbuat terhadap barang itu secara bebas sepenuhnya, asalkan tidak bertentangan dengan undang-undang atau peraturan umum...

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Adanya barang yang dimiliki | Tanah jaminan memang merupakan objek kebendaan | Tersedia dalam fakta |
| 2 | Hak menikmati dan berbuat secara bebas | ❌ Kasus tidak membahas pelaksanaan hak milik atas barang, melainkan sengketa wanprestasi kredit | Inti kasus adalah hubungan perikatan, bukan pelaksanaan hak kebendaan |

**KESIMPULAN**: Unsur pokok Pasal 570 **tidak terpenuhi**. Kasus ini bukan tentang pelaksanaan atau pelanggaran hak milik, melainkan tentang wanprestasi dalam perjanjian kredit. → **TIDAK RELEVAN**

---

## Contoh Kasus Kedua: Perbuatan Melawan Hukum — Pengangkutan Barang Rusak

### Deskripsi Kasus

> Penggugat menyewa jasa Tergugat untuk mengangkut perangkat server bernilai tinggi menggunakan layanan pengangkutan bersama tanpa memberitahukan sifat barang yang mudah rusak atau memerlukan pengemasan khusus. Barang diserahkan dalam kondisi tertutup rapat tanpa catatan kerusakan saat tanda tangan, namun ditemukan rusak saat tiba di tujuan. Tergugat menolak tanggung jawab karena telah menyelesaikan kewajiban pengangkutan sesuai instruksi dan barang diterima dengan baik oleh penerima, sementara Penggugat dianggap lalai dalam pengemasan. Hakim menolak gugatan ganti rugi karena tidak dapat dibuktikan bahwa kerusakan disebabkan oleh perbuatan Tergugat.

### Analisis Kasus Kedua

#### Analisis terhadap Pasal 1365 — Perbuatan Melawan Hukum

> **Pasal 1365**: Tiap perbuatan yang melanggar hukum dan membawa kerugian kepada orang lain, mewajibkan orang yang menimbulkan kerugian itu karena kesalahannya untuk menggantikan kerugian tersebut.

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Adanya perbuatan | Tergugat melakukan pengangkutan barang milik Penggugat | Tergugat melaksanakan jasa pengangkutan |
| 2 | Perbuatan melanggar hukum | Penggugat mendalilkan bahwa pengangkutan dilakukan tanpa kehati-hatian yang memadai | Dalil yang diajukan oleh Penggugat dalam gugatan |
| 3 | Adanya kerugian | Perangkat server ditemukan rusak saat tiba di tujuan | Kerugian materiil berupa kerusakan barang bernilai tinggi |
| 4 | Adanya kesalahan | ❓ Hakim tidak dapat membuktikan bahwa kerusakan disebabkan oleh perbuatan Tergugat; Penggugat dianggap lalai dalam pengemasan | Beban pembuktian tidak terpenuhi |
| 5 | Hubungan kausal | ❓ Tidak terbukti hubungan kausal antara perbuatan Tergugat dan kerusakan barang | Kerusakan bisa disebabkan pengemasan yang tidak memadai oleh Penggugat |

**KESIMPULAN**: Unsur (1), (2), (3) terpenuhi secara faktual, namun unsur (4) dan (5) — kesalahan dan hubungan kausal — **tidak terbukti**. Meskipun demikian, Pasal 1365 tetap **RELEVAN** karena pasal ini merupakan dasar hukum yang digunakan untuk menilai gugatan PMH, walaupun pada akhirnya hakim menolak gugatan karena unsur pembuktian tidak terpenuhi. *Relevansi pasal ditentukan oleh apakah unsur-unsurnya dibahas dalam kasus, bukan apakah unsur tersebut terbukti.*

---

#### Analisis terhadap Pasal 1366 — Kelalaian

> **Pasal 1366**: Setiap orang bertanggung jawab, bukan hanya atas kerugian yang disebabkan perbuatan-perbuatan, melainkan juga atas kerugian yang disebabkan kelalaian atau kesembronoannya.

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Adanya kelalaian atau kesembronoan | Penggugat mendalilkan bahwa Tergugat lalai dalam menangani barang rapuh; di sisi lain, Tergugat mendalilkan bahwa Penggugat lalai dalam pengemasan | Kedua pihak saling mendalilkan kelalaian |
| 2 | Adanya kerugian | Perangkat server rusak | Kerugian materiil terbukti |
| 3 | Hubungan kausal | Diperdebatkan — kerusakan bisa akibat kelalaian pengangkut atau kelalaian pengirim | Hakim menilai Penggugat yang lalai |

**KESIMPULAN**: Unsur-unsur Pasal 1366 **dibahas** dalam kasus ini, khususnya soal kelalaian kedua pihak. → **RELEVAN**

---

#### Analisis terhadap Pasal 1367 — Tanggung Jawab atas Bawahan (contoh TIDAK RELEVAN)

> **Pasal 1367**: Seseorang tidak hanya bertanggung jawab atas kerugian yang disebabkan perbuatannya sendiri, melainkan juga atas kerugian yang disebabkan perbuatan orang-orang yang menjadi tanggungannya...

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Hubungan subordinasi/pengawasan | ❌ Kasus ini melibatkan hubungan kontraktual jasa pengangkutan antara dua pihak setara, bukan hubungan majikan-bawahan atau orangtua-anak | Tidak ada fakta yang menunjukkan Tergugat bertanggung jawab atas perbuatan pihak ketiga |

**KESIMPULAN**: Unsur pokok Pasal 1367 (hubungan subordinasi) **tidak ada** dalam fakta kasus. → **TIDAK RELEVAN**

---

## Contoh Kasus Ketiga: Perjanjian Pernikahan dan Itikad Buruk

### Deskripsi Kasus

> [Tergugat] mengajukan gugatan rekonvensi untuk membatalkan Akta Perjanjian Pernikahan yang dibuatnya karena merasa dipaksa dan dibuat dengan itikad buruk oleh [Penggugat]. Namun, majelis hakim menyatakan bahwa sengketa pembatalan perjanjian tersebut harus ditangani dalam gugatan terpisah dan tidak dapat digabungkan dengan perkara perceraian. Akibatnya, gugatan rekonvensi tersebut dinyatakan tidak dapat diterima oleh pengadilan.

### Analisis Kasus Ketiga

#### Analisis terhadap Pasal 1320 — Syarat Sah Perjanjian

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Kesepakatan para pihak | Tergugat mendalilkan bahwa kesepakatan diberikan di bawah tekanan/paksaan — cacat kehendak | Paksaan dapat membatalkan perjanjian (lihat juga Pasal 1321-1328) |
| 2 | Kecakapan para pihak | Kedua pihak adalah suami-istri yang cakap hukum | Tidak dipersoalkan dalam kasus |
| 3 | Suatu pokok persoalan tertentu | Objek perjanjian: pisah harta dalam pernikahan | Objek jelas dan tertentu |
| 4 | Sebab yang tidak terlarang | Tergugat mendalilkan itikad buruk Penggugat — mempertanyakan kausa perjanjian | Itikad buruk dapat mempengaruhi keabsahan kausa |

**KESIMPULAN**: Unsur-unsur Pasal 1320 menjadi pokok sengketa, khususnya unsur (1) kesepakatan dan (4) kausa. → **RELEVAN**

---

#### Analisis terhadap Pasal 1338 — Kekuatan Mengikat Perjanjian

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Persetujuan yang dibuat sesuai UU | Akta Perjanjian Pernikahan dibuat secara formal (dengan akta) | Perjanjian dibuat dalam bentuk akta |
| 2 | Berlaku sebagai UU bagi para pihak | Perjanjian pisah harta mengikat kedua pihak selama perkawinan | Kekuatan mengikat dipertanyakan karena dalil paksaan |
| 3 | Pelaksanaan dengan itikad baik | Tergugat mendalilkan Penggugat tidak beritikad baik dalam pembuatan perjanjian | Inti sengketa menyangkut itikad baik |

**KESIMPULAN**: Pasal 1338 relevan karena kasus membahas kekuatan mengikat perjanjian dan itikad baik. → **RELEVAN**

---

#### Analisis terhadap Pasal 1243 — Ganti Rugi atas Wanprestasi

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Perikatan yang tidak dipenuhi | Tergugat mendalilkan bahwa perjanjian pernikahan tidak dilaksanakan sebagaimana mestinya | Keterkaitan dengan pelaksanaan perjanjian pisah harta |
| 2 | Debitur dinyatakan lalai | ❌ Tidak ada fakta tentang somasi atau pernyataan lalai | Kasus bukan tentang kelalaian memenuhi perikatan |

**KESIMPULAN**: Unsur somasi/pernyataan lalai **tidak terpenuhi**. Kasus ini tentang pembatalan perjanjian, bukan wanprestasi. → **TIDAK RELEVAN**

---

## Contoh Kasus Keempat: Sengketa Tanah dan Pemalsuan Sertifikat

### Deskripsi Kasus

> Penggugat mengajukan gugatan perdata terhadap beberapa pejabat dan perusahaan, menuduh mereka memalsukan surat bupati dan sertifikat hak guna usaha (HGU) serta melakukan korupsi atas tanah negara. Penggugat meminta pembatalan sertifikat dan pembayaran ganti rugi besar. Tergugat I mengajukan keberatan bahwa kasus ini menyangkut dugaan tindak pidana dan pembatalan administrasi yang seharusnya ditangani oleh pengadilan pidana atau tata usaha negara, bukan pengadilan perdata. Majelis hakim menerima keberatan tersebut dan menyatakan bahwa inti sengketa adalah perbuatan melawan hukum dalam konteks hukum pidana, sehingga pengadilan perdata tidak berwenang mengadili perkara ini.

### Analisis Kasus Keempat

#### Analisis terhadap Pasal 1365 — Perbuatan Melawan Hukum

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Adanya perbuatan | Pemalsuan surat bupati dan sertifikat HGU oleh para Tergugat | Perbuatan nyata yang didalilkan |
| 2 | Perbuatan melanggar hukum | Pemalsuan dokumen dan korupsi tanah negara | Melanggar hukum pidana dan hukum administrasi |
| 3 | Adanya kerugian | Penggugat menuntut ganti rugi besar atas hilangnya hak atas tanah | Kerugian materiil yang didalilkan |
| 4 | Adanya kesalahan | Pemalsuan dilakukan secara sengaja oleh para Tergugat | Kesengajaan dalam memalsukan dokumen |
| 5 | Hubungan kausal | Pemalsuan sertifikat menyebabkan Penggugat kehilangan hak atas tanah | Hubungan sebab-akibat yang jelas |

**KESIMPULAN**: Secara unsur, seluruh elemen Pasal 1365 **dibahas** dalam kasus. Meskipun hakim menyatakan pengadilan perdata tidak berwenang (karena inti sengketa bersifat pidana), pasal ini tetap **RELEVAN** sebagai dasar dalil gugatan perdata yang diajukan Penggugat. → **RELEVAN**

---

#### Analisis terhadap Pasal 1320 — Syarat Sah Perjanjian

| No | Unsur Hukum Perdata | Fakta Kasus yang Memenuhi Unsur | Dasar Pertimbangan |
|----|---------------------|--------------------------------|--------------------|
| 1 | Kesepakatan para pihak | ❌ Kasus tidak membahas adanya perjanjian antara Penggugat dan Tergugat | Sengketa bukan tentang hubungan kontraktual |

**KESIMPULAN**: Kasus ini bukan tentang keabsahan perjanjian, melainkan tentang perbuatan melawan hukum. Tidak ada unsur perjanjian yang dibahas. → **TIDAK RELEVAN**

> **Catatan**: Pada ground truth saat ini, kasus ini justru dipetakan ke Pasal 1320 dan 1338. Hal ini menunjukkan potensi **mislabel** dalam ground truth yang bersumber dari kutipan hakim — hakim mungkin mengutip pasal-pasal tersebut sebagai referensi umum tanpa relevansi substantif terhadap fakta kasus.

---

## Panduan Penilaian Relevansi

### Kapan Pasal Dinyatakan RELEVAN

Pasal dinyatakan **RELEVAN** jika:

1. **Unsur-unsur pasal dibahas** dalam fakta kasus — baik yang terbukti maupun yang diperdebatkan
2. Pasal menjadi **dasar dalil** salah satu pihak (penggugat atau tergugat)
3. Pasal menjadi **dasar pertimbangan** hakim dalam putusannya
4. Fakta kasus **memenuhi atau menyentuh** unsur-unsur pokok pasal, meskipun hakim akhirnya menolak gugatan

### Kapan Pasal Dinyatakan TIDAK RELEVAN

Pasal dinyatakan **TIDAK RELEVAN** jika:

1. **Unsur pokok** pasal sama sekali tidak ada dalam fakta kasus
2. Fakta kasus membahas **domain hukum yang berbeda** (misalnya kasus wanprestasi dianalisis terhadap pasal pewarisan)
3. Hanya ada kesamaan **kata kunci** permukaan tanpa keterkaitan substansial (misalnya kata "tanah" muncul di kasus dan di pasal tentang kebendaan, tetapi konteks hukumnya berbeda)

### Perbedaan dengan Ground Truth Berbasis Kutipan

| Aspek | Ground Truth Kutipan (saat ini) | Ground Truth Rubrik (yang diusulkan) |
|-------|--------------------------------|-------------------------------------|
| **Sumber** | Pasal yang dikutip hakim/penggugat dalam putusan | Penilaian unsur-per-unsur terhadap fakta kasus |
| **Kelemahan** | Hub bias — pasal umum (1365, 1320, 1338) dikutip berlebihan sebagai "boilerplate" | Lebih mahal secara komputasi |
| **Kelebihan** | Ground truth objektif dari praktisi hukum | Mengidentifikasi pasal yang relevan tapi tidak dikutip |
| **Contoh kesalahan** | Pasal 1320 dikutip di kasus PMH tanpa konteks perjanjian | Rubrik akan menolak karena unsur perjanjian tidak terpenuhi |

---

## Integrasi ke Pipeline Ekspansi Ground Truth

Rubrik ini akan digunakan untuk memperbaiki prompt LLM di `src/data/expand_qrels.py`:

### Prompt Saat Ini (Sederhana)
```
Jawab untuk setiap pasal: RELEVAN atau TIDAK_RELEVAN.
- RELEVAN: fakta dalam kasus memenuhi atau membahas unsur-unsur pasal tersebut.
- TIDAK_RELEVAN: fakta dalam kasus tidak menyentuh unsur-unsur pasal tersebut.
```

### Prompt yang Diusulkan (Berbasis Rubrik)
```
Untuk setiap pasal, lakukan analisis unsur-per-unsur:
1. Identifikasi unsur-unsur pokok dari pasal tersebut
2. Periksa apakah setiap unsur pokok terpenuhi atau dibahas oleh fakta kasus
3. Jika SELURUH unsur pokok terpenuhi/dibahas → RELEVAN
4. Jika ada unsur pokok yang sama sekali tidak ada dalam fakta → TIDAK_RELEVAN

Penting:
- "Dibahas" berarti fakta kasus menyentuh unsur tersebut, meskipun hakim menolak
- Jangan menilai RELEVAN hanya karena ada kesamaan kata kunci permukaan
- Pasal umum (1365, 1320, 1338) harus dianalisis seketat pasal khusus
```

---

## Lampiran: Perbandingan Format Rubrik Pidana vs Perdata

### Format Rubrik Pidana (https://aclc.kpk.go.id/materi-pembelajaran/hukum/buku/buku-saku-memahami-untuk-membasmi)

```
Pasal 2 UU No. 31 Tahun 1999 jo. UU No. 20 Tahun 2001:
(1) Setiap orang yang secara melawan hukum melakukan perbuatan memperkaya
    diri sendiri atau orang lain atau suatu korporasi yang dapat merugikan
    keuangan negara atau perekonomian negara...

| No | Unsur Tindak Pidana        | Fakta Perbuatan               | Alat Bukti            |
|----|---------------------------|-------------------------------|-----------------------|
| 1  | Setiap orang              | B adalah seorang Dirut BUMN   | Keterangan Terdakwa B |
| 2  | Memperkaya diri sendiri   | B mendapat transfer Rp 15M    | Rekening bank         |
| 3  | Dengan cara melawan hukum | B menjual aset di bawah NJOP  | Keterangan Saksi      |
| 4  | Dapat merugikan keuangan  | Negara dirugikan Rp 50M       | Laporan BPKP          |

KESIMPULAN: Keempat unsur terpenuhi → terbukti tindak pidana korupsi.
```

### Format Rubrik Perdata (Adaptasi)

```
Pasal 1365 KUHPerdata:
Tiap perbuatan yang melanggar hukum dan membawa kerugian kepada orang lain,
mewajibkan orang yang menimbulkan kerugian itu karena kesalahannya untuk
menggantikan kerugian tersebut.

| No | Unsur Hukum Perdata     | Fakta Kasus yang Memenuhi   | Dasar Pertimbangan        |
|----|------------------------|-----------------------------|---------------------------|
| 1  | Adanya perbuatan       | Tergugat melakukan X        | Fakta dalam posita        |
| 2  | Melanggar hukum        | Perbuatan X melanggar Y     | Norma yang dilanggar      |
| 3  | Adanya kerugian        | Penggugat mengalami Z       | Bukti kerugian materiil   |
| 4  | Adanya kesalahan       | Tergugat sengaja/lalai      | Dalil penggugat           |
| 5  | Hubungan kausal        | Perbuatan X menyebabkan Z   | Kausalitas fakta          |

KESIMPULAN: Seluruh unsur terpenuhi/dibahas → RELEVAN.
```
