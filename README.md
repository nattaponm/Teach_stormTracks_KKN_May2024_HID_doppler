# Teach_stormTracks_KKN_May2024_HID_doppler

สอนวิเคราะห์เส้นทาง แหล่งกำเนิด พัฒนาการ ความรุนแรง และความถี่เชิงพื้นที่ของพายุ **ทั้งเดือนพฤษภาคม 2567** จากเรดาร์ขอนแก่น (KKN) กรมอุตุนิยมวิทยา ร่วมกับกายภาพเมฆจาก reflectivity, Doppler, spectrum width, ตัวแปร dual-polarization และการจำแนกชนิดหยาดน้ำฟ้า (HID) เทียบกับฐานข้อมูลความเสียหายรายอำเภอและภูมิประเทศ (DEM)

พฤษภาคมเป็นช่วงคาบเกี่ยวระหว่างฤดูร้อน (pre-monsoon) กับฤดูฝน (กรมอุตุนิยมวิทยาประกาศเข้าสู่ฤดูฝน 21 พ.ค. 2567) ชุดบทเรียนนี้เป็นภาคต่อของ
[Teach_Pysteps_KKN_summerStorm](https://github.com/nattaponm/Teach_Pysteps_KKN_summerStorm),
[Teach_HID_Doppler_KKN_summerStorm](https://github.com/nattaponm/Teach_HID_Doppler_KKN_summerStorm) และ
[Teach_Tdating_strom_tracks_HID_CloudProperties](https://github.com/nattaponm/Teach_Tdating_strom_tracks_HID_CloudProperties) ซึ่งศึกษาพายุวันเดียว (3 พ.ค. 2567)

ผู้เรียน: นิสิตปริญญาตรีปี 3–4 และปริญญาโท ด้านวิทยาศาสตร์สิ่งแวดล้อม ภูมิศาสตร์ และอุตุนิยมวิทยา ทุก notebook เป็นภาษาไทย มีวัตถุประสงค์การเรียนรู้ สมการ แผนที่มาตรฐานวารสาร แบบฝึกหัด 3 ข้อพร้อมเฉลย คำถามชวนคิด และแนวทางต่อยอด

## บทเรียน (`notebooks/`)

| บท | เนื้อหา | สถานะ |
|---|---|---|
| 01 | ฐานข้อมูลภัยพิบัติ เรขาคณิตเรดาร์ ดัชนีความรุนแรง จัดกลุ่มวัน | ✅ |
| 02 | จาก CAPPI สู่วัตถุพายุ: ภาพรวมรายวันทั้งเดือน | รอข้อมูล derived |
| 03 | T-DaTing กับการทดลอง Δt = 15 / 10 / 5 นาที | รอข้อมูล derived |
| 04 | Cloud footprint และ zonal statistics กับ Z, Doppler, dual-pol, HID | รอข้อมูล derived |
| 05 | ภูมิประเทศ ระยะเรดาร์ และการตรวจสอบกับรายงานความเสียหาย | รอข้อมูล derived |
| 06 | ความถี่เชิงพื้นที่ สถิติเชิงพื้นที่ สรุปรายตำบล/อำเภอ/จังหวัด | รอข้อมูล derived |

## การเตรียมข้อมูล (`prep/`, สำหรับผู้ดูแลข้อมูล)

| ขั้น | เนื้อหา | สถานะ |
|---|---|---|
| P0 | สำรวจไฟล์ UF ทั้งเดือน: ช่องเวลา ไฟล์เสีย Nyquist พิกัดเรดาร์ ความพร้อมรายวัน | ✅ |
| P1 | CAPPI ทุกวัน + ดัชนีความเคลื่อนไหวของพายุต่อ volume | ถัดไป |
| P2 | Dealiasing, KDP, ZDR, HID, divergence, azimuthal shear, EDR → กริด 1 km | ถัดไป |
| P3 | วัตถุ เส้นทาง (15/10/5 นาที) zonal stats และสรุปทั้งเดือน | ถัดไป |
| P4 | จัดแพ็ก ตรวจ checksum อัปโหลด Zenodo (draft) | ถัดไป |

ลำดับการรัน: **บท 01 → P0 → บท 01 อีกครั้ง → P1 → P2 → P3 → P4** แล้วนิสิตเรียน 01–06

## นโยบายข้อมูล

- **ไฟล์ UF ของกรมอุตุนิยมวิทยาไม่เผยแพร่** เก็บใน Google Drive ส่วนตัวของผู้ดูแลเท่านั้น โค้ดใน `prep/` เผยแพร่เพื่อความโปร่งใสของวิธีการ แต่รันได้เฉพาะผู้มีสิทธิ์เข้าถึงข้อมูล
- สิ่งที่เผยแพร่คือผลิตภัณฑ์ที่ประมวลผลแล้วบนกริด Cartesian 1 km (quantized) และตารางสรุป ไม่มีข้อมูลในพิกัด polar
- ใน repo: ตารางทั้งเดือนและกริดของวันหลัก (≤ 80 MB) ทำให้ทุกบทรันได้ทันที / ใน Zenodo (DOI): กริดรายวันทั้งเดือน

## โครงสร้าง

```
config/settings.yaml      ค่าตั้งต้นทั้งหมด (เรดาร์ กริด ช่วงของเดือน กติกาข้อมูล)
config/radar_site.json    พิกัดเรดาร์จริงจาก UF (สร้างโดย P0)
data/                     Excel รายงาน, gis/ (ขอบเขตปกครอง, DEM), reports/, inventory/
utils/                    config, geo, reports, maps, radar_inventory
assets/fonts/             ฟอนต์ Sarabun (SIL OFL 1.1) สำหรับรูปภาษาไทย
```

## การใช้งานบน Colab

เปิด notebook จาก GitHub ใน Colab แล้วรันเซลล์แรก ระบบจะ clone repo และติดตั้งไลบรารีให้ ครั้งแรกที่รันบท 01 จะสร้าง DEM 1 km จาก Copernicus GLO-90 อัตโนมัติ (2–5 นาที) ผู้ดูแลควร commit `data/gis/dem_kkn_1km.tif` เข้า repo หลังรันครั้งแรก

## แหล่งข้อมูลและสัญญาอนุญาต

- ข้อมูลเรดาร์: กรมอุตุนิยมวิทยา (ใช้ภายใต้ข้อตกลง ไม่เผยแพร่ข้อมูลดิบ)
- ฐานข้อมูลความเสียหาย: รวบรวมจาก ปภ. และสำนักงานประชาสัมพันธ์จังหวัด
- ขอบเขตการปกครอง: [OpenGISData-Thailand](https://github.com/chingchai/OpenGISData-Thailand) (CC BY 4.0)
- DEM: Copernicus GLO-90 © DLR e.V. 2010–2014 และ © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA
- ฟอนต์ Sarabun: SIL Open Font License 1.1
