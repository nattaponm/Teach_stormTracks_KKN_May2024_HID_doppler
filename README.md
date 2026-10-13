# Teach_stormTracks_KKN_May2024_HID_doppler

สอนวิเคราะห์ **เส้นทาง แหล่งกำเนิด พัฒนาการ ความรุนแรง และความถี่เชิงพื้นที่ของพายุทั้งเดือนพฤษภาคม 2567** จากเรดาร์ขอนแก่น (KKN) กรมอุตุนิยมวิทยา ร่วมกับกายภาพเมฆจาก reflectivity, Doppler, spectrum width/EDR, dual-polarization และการจำแนกชนิดหยาดน้ำฟ้า (fuzzy HID) เทียบกับฐานข้อมูลความเสียหายรายอำเภอ ภูมิประเทศ (DEM) และขอบเขตตำบล/อำเภอ/จังหวัด

พฤษภาคมคาบเกี่ยวฤดูร้อน (pre-monsoon) กับฤดูฝน (กรมอุตุนิยมวิทยาประกาศเข้าฤดูฝน 21 พ.ค. 2567) จึงแบ่งเดือนเป็น 3 ช่วง: **ระลอกพายุฤดูร้อน 1–9**, **ปลายฤดูร้อน 10–20**, **ต้นฤดูฝน 21–31 พ.ค.** วิเคราะห์ **ทุกวัน** รวมวันที่ไม่มีรายงานความเสียหาย (ใช้เป็นกลุ่มเปรียบเทียบ และลดอคติจากการรายงาน)

ภาคต่อของ [Teach_Pysteps_KKN_summerStorm](https://github.com/nattaponm/Teach_Pysteps_KKN_summerStorm),
[Teach_HID_Doppler_KKN_summerStorm](https://github.com/nattaponm/Teach_HID_Doppler_KKN_summerStorm) และ
[Teach_Tdating_strom_tracks_HID_CloudProperties](https://github.com/nattaponm/Teach_Tdating_strom_tracks_HID_CloudProperties) (พายุวันเดียว 3 พ.ค. 2567) — ใช้โค้ดและพารามิเตอร์เดิมเป็นฐาน

ผู้เรียน: นิสิตปริญญาตรีปี 3–4 และปริญญาโท สาขาวิทยาศาสตร์สิ่งแวดล้อม ภูมิศาสตร์ และอุตุนิยมวิทยา ทุกบทเป็นภาษาไทย มีวัตถุประสงค์ แนวคิด สมการ แผนที่คุณภาพวารสาร แบบฝึกหัดพร้อมเฉลย คำถามชวนคิด และแนวทางต่อยอดเป็นงานวิจัย

## บทเรียนสำหรับนิสิต (`notebooks/`)

| บท | เนื้อหา | เปิดใน Colab |
|---|---|---|
| 01 | ฐานข้อมูลความเสียหาย เรขาคณิตเรดาร์ 240 km ดัชนีความรุนแรง จัดกลุ่มวันและช่วงของเดือน | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/01_reports_radar_geometry_day_selection.ipynb) |
| 02 | จาก CAPPI 2 km สู่วัตถุพายุ: ปฏิทินพายุทั้งเดือน วัฏจักรรายวัน ความถี่ ≥ 35 dBZ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/02_cappi_echo_objects_month.ipynb) |
| 03 | T-DaTing และ advection interpolation: เปรียบเทียบเส้นทาง Δt = 15 / 10 / 5 นาที | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/03_tdating_tracks_dt15_10_5.ipynb) |
| 04 | Cloud footprint, zonal statistics, boxplot Z/Doppler/dual-pol/HID/EDR ตามกลุ่มอายุ + การทดสอบสถิติ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/04_footprints_zonal_radar_variables.ipynb) |
| 05 | ภูมิประเทศ (DEM) แหล่งกำเนิด ระยะเรดาร์ (ตัวกวน) จับคู่เส้นทางกับรายงานความเสียหาย | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/05_terrain_range_reports.ipynb) |
| 06 | แผนที่ความถี่ KDE Moran's I, Getis–Ord Gi* สรุปรายตำบล/อำเภอ/จังหวัด ส่งออก GIS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/06_spatial_frequency_admin.ipynb) |
| 07 | พายุที่มีรายงานความเสียหาย vs ไม่มีรายงาน: ระดับความเสียหายจากจำนวนบ้าน (Weak/Moderate/Severe) dose–response, matched case–control, การสัมผัสภัย (GHSL) การบังลำคลื่น | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/notebooks/07_damage_severity_vs_radar.ipynb) |

นิสิตไม่ต้องเข้าถึงข้อมูลเรดาร์ดิบ: ทุกบทอ่านตารางทั้งเดือนและกริดของวันหลักจาก `data/` ใน repo ส่วนวันอื่นดาวน์โหลดจาก Zenodo อัตโนมัติผ่าน `utils.data.day_file()` (ตรวจ sha256)

## การเตรียมข้อมูล (`prep/`, สำหรับผู้ดูแลข้อมูลเท่านั้น)

| ขั้น | เนื้อหา | เปิดใน Colab |
|---|---|---|
| P0 | สำรวจ UF ทั้งเดือน: ช่องเวลา 15 นาที ไฟล์เสีย การบีบอัด (gzip/bz2) Nyquist scan strategy ช่วงข้อมูลขาด | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/P0_radar_inventory.ipynb) |
| PS | โปรไฟล์อุณหภูมิรายวันสำหรับ HID: โคราช 48431 → อุบล 48407 → เฉลี่ยวันข้างเคียง + ดัชนีสภาพแวดล้อม | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/PS_sounding_hid_profiles.ipynb) |
| P1 | pseudo-CAPPI 2 km + Z สูงสุดในคอลัมน์ ทุก volume ทุกวัน (ตรวจเทียบ pysteps) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/P1_cappi_all_days.ipynb) |
| P2 | Dealias, DIV, AZSHEAR, KDP, ZDR, Z-PHI, HID (Besic 2016), EDR เฉพาะช่วงมีพายุ → กริด 1 km (ตรวจเทียบ repo3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/P2_polar_products_hid.ipynb) |
| P3 | แบ่งเหตุการณ์พายุทั้งเดือน → เส้นทาง T-DaTing 15/10/5 นาที (ต่อเนื่องข้ามวัน) footprint ตารางเซลล์/เส้นทาง สรุปกริดทั้งเดือน (ตรวจเทียบ repo3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/P3_tracking_month.ipynb) |
| P4 | checksum เลือกวันหลักเข้า repo (≤ 80 MB) สร้าง data_sources.yaml อัปโหลด Zenodo (draft) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/P4_package_release.ipynb) |
| P5 | GHSL (ประชากร/สิ่งปลูกสร้าง 1 km) เข้ากริดเรดาร์ DEM ค่าสูงสุด → beam blockage fraction (Bech 2003) ตรวจเทียบความถี่พายุตามมุม การสัมผัสภัยรายอำเภอ/รายเส้นทาง | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattaponm/Teach_stormTracks_KKN_May2024_HID_doppler/blob/main/prep/P5_exposure_blockage.ipynb) |

**ลำดับการรัน (Colab Pro, CPU + High-RAM):** P0 → บท 01 (`ADMIN_SYNC = True`) → PS → P1 → P2 → P3 → P4 → P5 (`COMMIT = True`) แล้วจึงให้นิสิตเรียนบท 01–07
เวลาโดยประมาณ: P1 ไม่กี่นาที/วัน · P2 ≈ 11 วินาที/volume/core (เฉพาะช่วงมีพายุ) · P3 ≈ 1 ชม. ทั้งเดือน (8 core) · P5 ≈ 10–15 นาที (ดาวน์โหลด GHSL ~450 MB ครั้งแรก เก็บ cache ใน Drive) ทุกขั้น **resume ได้** (ข้ามวันที่มีไฟล์ผลลัพธ์แล้ว) ผลลัพธ์ระหว่างทางเก็บใน Drive ที่ `radar_data.drive_derived_root` ใน `config/settings.yaml`

### Colab แยกเครื่องต่อ notebook → ส่งต่อผลผ่าน Drive (`utils/sync.py`)
แต่ละ notebook ใน Colab รันบนเครื่องเสมือนของตัวเอง ไฟล์ที่ P0 เขียนลงโฟลเดอร์ repo จึงไม่มีในเครื่องของบท 01 หรือ P4 และหายเมื่อปิด runtime ทุก notebook ใน `prep/` จึง
- **ต้น notebook:** `sync.pull(OUT_ROOT)` ดึงผลของขั้นก่อนหน้าจาก `KKN_derived_v1/repo_mirror/` ใน Drive
- **ท้าย notebook:** `sync.push(OUT_ROOT)` เก็บไฟล์ที่จะเข้า repo (`data/inventory`, `data/reports`, `data/sounding`, `data/derived`, `data/days`, `config/radar_site.json`, `config/data_sources.yaml`, DEM) กลับเข้า Drive — **รันเซลล์สุดท้ายทุกครั้งก่อนปิด runtime**
- **commit:** ตั้ง `COMMIT = True` ในเซลล์สุดท้าย (แนะนำที่ P4 ครั้งเดียว) ใช้ fine-grained token ใน Colab Secrets ชื่อ `KKN_commit` (สิทธิ์ Contents: Read and write เฉพาะ repo นี้) — `sync.git_push` กรอง token ออกจากทุกข้อความที่พิมพ์
- **บท 01 (ผู้ดูแล):** ตั้ง `ADMIN_SYNC = True` ในเซลล์ 🔒 เพื่อดึงผล P0 และเก็บ `data/reports` เข้า Drive (นิสิตปล่อยเป็น False)

### Sounding สำหรับ HID (PS)
HID ต้องการโปรไฟล์อุณหภูมิถึง ≳ 12 km หากใช้โปรไฟล์ที่สั้น `np.interp` จะตรึงอุณหภูมิบนสุดไว้ทั้งคอลัมน์ ทำให้ชั้นน้ำแข็งหายไป กติการายวัน:
1. **โคราช (48431)** 00 UTC → ถ้าไม่ได้ ลอง 12 UTC วันก่อน / แหล่ง BUFR–FM35
2. ถ้าสูงไม่ถึง 12 km แต่ใช้ได้ส่วนล่าง → **splice** ต่อด้วย composite ทั้งเดือน (offset ค่อย ๆ ลดลงใน 2 km)
3. ถ้าโคราชใช้ไม่ได้ → **อุบลราชธานี (48407)**
4. ถ้าอุบลก็ไม่มี → **ค่าเฉลี่ยของวันข้างเคียง (±3 วัน)** ที่มีโปรไฟล์สมบูรณ์
5. สุดท้าย → composite ทั้งเดือน

ทุกวันมีป้าย `source` ใน `data/sounding/may2024_hid_profiles.nc` และตาราง `may2024_hid_profile_summary.csv` ทดสอบความไวกับ volume 3 พ.ค. 09:30 UTC: ใช้ค่าเฉลี่ยวันข้างเคียงเปลี่ยนชนิด HID 4.6% ของ gate, composite 1.4%, ส่วนโปรไฟล์สั้น 1.1 km เปลี่ยน 42.7% (ไม่มีคลาสน้ำแข็งเลย) — จึงห้ามใช้โปรไฟล์สั้นโดยไม่ splice

### การตรวจสอบกับ repo เดิม (3 พ.ค. 2567)
| ขั้น | เทียบกับ | ผล |
|---|---|---|
| P1 pseudo-CAPPI 2 km | Teach_Pysteps (pysteps) | ต่างเฉลี่ย 0.00 dBZ |
| P2 ตัวแปร polar บนกริด | Teach_Tdating (repo3) | ตรงกัน 100% |
| P3 เส้นทาง T-DaTing 15 นาที | repo3 | 115 / 59 เทียบ 114 / 59 (วัตถุ/เส้นทาง) |

หมายเหตุ: PhiDP ของ KKN ใช้เครื่องหมาย **−1 คงที่** (`PHIDP_SIGN` ใน `utils/radar_prep.py`) เพราะการประมาณเครื่องหมายอัตโนมัติจาก slope ให้ NaN แล้วตกเป็น +1 ในบาง volume

### พารามิเตอร์หลัก
- กริด AEQD 1 km 481×481 รัศมี 240 km รอบเรดาร์ (พิกัดจริงจาก UF → `config/radar_site.json`)
- T-DaTing (เหมือน repo3): minref 33, maxref 45, mindiff 5, minsize 15 km², minmax 40, mindis 8 km, ติดตามเป็น **เหตุการณ์พายุ (episode)** ไม่ใช่ทีละวัน: ต่อข้อมูลทั้งเดือนแล้วตัดเฉพาะที่ช่วงสงบ (พื้นที่ ≥ 35 dBZ < 20 km² ติดกัน ≥ 1 ชม.) หรือข้อมูลขาด เส้นทางจึงต่อเนื่องข้ามเที่ยงคืน UTC (07:00 น.) ได้ (ทดสอบด้วยข้อมูลเลื่อนเวลาให้คร่อมเที่ยงคืน: ตรงกับการติดตามต่อเนื่อง 164/164 เส้น) `uid` = `E###-####`, `day` = วันที่ UTC ของจุดเริ่ม, `day_local` = วันที่เวลาไทย, `censored` = แตะขอบเหตุการณ์ที่ติดข้อมูลขาด/ขอบเดือน
- Δt: `15min` (ข้อมูลจริง), `10min_linZ`, `05min_linZ` (advection interpolation)
- ตัวแปร polar: Zc, vel, div, azshear (s⁻¹), edr, kdp, zdr, HID 4 ชั้นความสูง, hail/graupel column

## นโยบายข้อมูล

- **ไฟล์ UF ของกรมอุตุนิยมวิทยาไม่เผยแพร่** เก็บใน Google Drive ส่วนตัวของผู้ดูแล โค้ด `prep/` เผยแพร่เพื่อความโปร่งใสของวิธีการ แต่รันได้เฉพาะผู้มีสิทธิ์เข้าถึงข้อมูล `.gitignore` กันไฟล์ UF ไว้แล้ว
- เผยแพร่เฉพาะผลิตภัณฑ์บนกริด Cartesian 1 km (quantized int16) และตารางสรุป ไม่มีข้อมูลพิกัด polar
- **ใน repo:** ตารางทั้งเดือน (`data/derived/`) + กริดของวันหลัก (`data/days/YYYYMMDD/`) รวม ≤ 80 MB
- **ใน Zenodo (DOI):** กริดรายวันครบทั้งเดือน ลิงก์และ checksum อยู่ใน `config/data_sources.yaml` (สร้างโดย P4)
- **GHSL (P5):** ประชากรและสิ่งปลูกสร้าง R2023A 1 km ของ European Commission JRC (CC BY 4.0) — Pesaresi & Politis (2023); Schiavina et al. (2023) ใน repo เก็บเฉพาะค่าที่แปลงเข้ากริดเรดาร์แล้ว (`data/derived/exposure_1km.nc`)
- **ระดับความเสียหาย (บท 07):** กติกาทั้งหมดอยู่ใน `config/damage_rules.yaml` (เกณฑ์จำนวนบ้าน รายการที่ตัดออก ยอดรวมระดับจังหวัด การยกระดับเป็น Severe พร้อมข้อความหลักฐาน) แก้ไฟล์นี้แล้วรันบท 07 ใหม่ได้โดยไม่ต้องแก้โค้ด

## โครงสร้าง

```
config/settings.yaml        ค่าตั้งต้นทั้งหมด (เรดาร์ กริด ช่วงของเดือน sounding tracking release)
config/radar_site.json      พิกัดเรดาร์จริงจาก UF (P0)
config/data_sources.yaml    ลิงก์ Zenodo + sha256 (P4)
data/ข้อมูลพายุอีสานปี2024.xlsx  ฐานข้อมูลความเสียหาย
data/reports/               รายงานที่ทำความสะอาดแล้ว + ตารางวัน (บท 01)
data/gis/                   ขอบเขตการปกครอง, DEM 1 km
data/sounding/              โปรไฟล์ HID รายวัน + ดัชนีสภาพแวดล้อม (PS)
data/inventory/             บัญชีไฟล์เรดาร์ (P0)
data/derived/               ตารางเซลล์/เส้นทาง footprint สรุปทั้งเดือน (P1–P4)
data/days/YYYYMMDD/         กริดรายวันของวันหลัก (P4)
utils/                      โมดูล: config geo maps reports radar_io radar_prep sounding tracking derived_io data analysis
                            + polar_tools track_tools data_io kkn_tools (คัดลอกจาก repo3)
assets/fonts/               ฟอนต์ Sarabun (SIL OFL 1.1)
```

## การใช้งานบน Colab

กดปุ่ม **Open in Colab** แล้วรันเซลล์แรก ระบบจะ clone repo และติดตั้งไลบรารีให้ ครั้งแรกที่รันบท 01 จะสร้าง DEM 1 km จาก Copernicus GLO-90 (2–5 นาที) ผู้ดูแลควร commit `data/gis/dem_kkn_1km.tif` หลังรันครั้งแรก รูปทั้งหมดบันทึกเป็น PNG 300 dpi + PDF ใน `figures/`

## แหล่งข้อมูลและสัญญาอนุญาต

- ข้อมูลเรดาร์: กรมอุตุนิยมวิทยา (ใช้ภายใต้ข้อตกลง ไม่เผยแพร่ข้อมูลดิบ)
- ข้อมูลหยั่งอากาศ: University of Wyoming upper-air archive และชุดข้อมูลจาก [Teaching_Sounding_SevereStorm_Korat](https://github.com/nattaponm/Teaching_Sounding_SevereStorm_Korat)
- ฐานข้อมูลความเสียหาย: รวบรวมจาก ปภ. และสำนักงานประชาสัมพันธ์จังหวัด
- ขอบเขตการปกครอง: [OpenGISData-Thailand](https://github.com/chingchai/OpenGISData-Thailand) (CC BY 4.0)
- DEM: Copernicus GLO-90 © DLR e.V. 2010–2014 และ © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA
- ไลบรารีหลัก: Py-ART, pysteps, PyTDA, MetPy, GeoPandas
- ฟอนต์ Sarabun: SIL Open Font License 1.1
