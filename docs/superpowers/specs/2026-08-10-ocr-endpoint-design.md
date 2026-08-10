# OCR endpoint — design

Datum: 2026-08-10
Stav: návrh k odsouhlasení

## Cíl

Přidat do služby `image-embeddings` endpoint, který z fotky přečte text a vrátí ho
i se souřadnicemi a jistotou. Konzumentem je **kukátko**, které chce postupně projet
celý archiv fotek a uložit texty pro fulltextové vyhledávání.

## Rozsah

**Uvnitř:**

- Nový endpoint `POST /ocr/image` pro jednu fotku.
- Tištěný text: cedule, nápisy, plakáty, transparenty, SPZ, skeny dokumentů, strojopis, razítka.
- Čeština včetně diakritiky, vedle toho zbytek latinky.
- Rozšíření `/health` o informaci, na čem OCR jede.
- Testy a měření propustnosti na boxu.

**Mimo:**

- **Ručně psané popisky.** Klasické OCR enginy českou kurzívu nepřečtou a VLM, který by to
  zvládl, se nevejde do zbývající VRAM. Samostatný úkol, až bude čím ho obsloužit.
- **Napojení kukátka.** Volající strana je samostatný úkol v jiném repu.
- **Dávkový endpoint** (více fotek v jednom requestu). Až kdyby měření ukázalo, že režie
  jednoho requestu na fotku je úzké hrdlo.
- **Rozpad `server.py` na moduly.** Nový kód dostane vlastní modul, stávající se nepřepisuje.

## Volba enginu

**RapidOCR 3.9.2 + PP-OCRv5, běh přes ONNX Runtime.**

- Rozpoznávací model `latin_PP-OCRv5_mobile_rec` [oficiálně pokrývá 47 jazyků latinky
  včetně češtiny](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec).
- RapidOCR je ten samý model **bez závislosti na PaddlePaddle** — čistě `onnxruntime`,
  který je ve venvu už kvůli InsightFace.
- Stejnou kombinaci nasadil Immich v PR [immich#23527](https://github.com/immich-app/immich/pull/23527)
  (`LATIN__PP-OCRv5_mobile`), tedy fotogalerie se stejným use-casem — dávkové OCR archivu.

Zamítnuto: **PaddleOCR nativně** (druhý deep-learning framework vedle torche, ~6 GB,
křehké verzování proti `numpy<2`), **EasyOCR** (horší přesnost, ~1,5 GB VRAM navíc,
projekt jen udržovaný).

## Zdroj modelů

Modely se stahují **z HuggingFace**, ne z ModelScope:

| Role | Repo | Soubor |
|---|---|---|
| Detekce | `PaddlePaddle/PP-OCRv5_mobile_det_onnx` | `inference.onnx` |
| Rozpoznání | `PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx` | `inference.onnx` + `inference.yml` |

Slovník znaků se vytáhne z `PostProcess.character_dict` v `inference.yml` do
`latin_dict.txt`; ověřeno, že obsahuje českou diakritiku (`č`, `ě`, `ř`, `š`, `ů`).

**Proč ne ModelScope, odkud RapidOCR stahuje ve výchozím stavu:** doména `modelscope.cn`
je blokovaná resolverem DNS4EU Protective (`86.54.11.1`), který síť používá. Nefiltrovaná
varianta resolveru doménu vrací normálně, takže jde o blok podle threat intelligence,
ne o geo ani kategorii — Baidu, Taobao ani HuggingFace blokované nejsou. Důvod označení
se nepodařilo dohledat. Bezpečnostní rozhodnutí sítě se obcházet nebude; HuggingFace nese
tytéž modely od oficiální PaddlePaddle organizace a je dostupný. Vedlejší přínos: služba
nemá žádnou runtime závislost na externím CDN.

Modely se stahují **jednou skriptem**, ukládají lokálně a RapidOCR se na ně nasměruje přes
`Det.model_path`, `Rec.model_path` a `Rec.rec_keys_path`. Za běhu se nikam nesahá, takže
výpadek zdroje nemůže shodit start služby ani request.

## API kontrakt

```
POST /ocr/image
Content-Type: multipart/form-data
  file            povinné, image/*
  min_confidence  volitelné, float, výchozí 0.5
```

Odpověď 200:

```json
{
  "text": "HOSTINEC U KOZÁKŮ\nPlán budíků",
  "blocks_count": 2,
  "blocks": [
    {"text": "HOSTINEC U KOZÁKŮ", "bbox": [120.0, 340.0, 880.0, 430.0], "confidence": 0.97},
    {"text": "Plán budíků",       "bbox": [210.0, 620.0, 540.0, 668.0], "confidence": 0.61}
  ],
  "min_confidence": 0.5,
  "lang": "latin",
  "model": "PP-OCRv5_mobile"
}
```

Rozhodnutí:

- **`bbox` je `[x_min, y_min, x_max, y_max]`** v pixelech původního obrázku. RapidOCR vrací
  čtyřbodový polygon kvůli natočenému textu; bereme z něj opsaný obdélník. Důvod: `/embed/face`
  už dnes vrací bbox v tomhle tvaru, takže kukátko nemusí umět dva formáty souřadnic.
  Kdyby bylo natočení potřeba, přidá se vedle toho `polygon` — zpětně kompatibilně.
- **Bloky ve čtecím pořadí** — bloky se seskupí do řádků podle středu na ose y s tolerancí
  poloviny mediánové výšky bloku, řádky se řadí shora dolů a uvnitř řádku zleva doprava.
  Bez té tolerance by mírně nakloněný nápis rozházel pořadí.
- **`model`** označuje dvojici det + rec modelu (`PP-OCRv5_mobile`), `lang` říká, která
  jazyková varianta rozpoznávacího modelu je načtená.
- **`text`** je spojení textů bloků přes `\n`.
- **Práh `min_confidence` platí stejně na `text` i na `blocks`**, aby nemohl nastat stav,
  kdy fulltext obsahuje něco, co v blocích není.
- **Fotka bez textu = 200** s `"text": ""` a prázdným polem. U dávkové indexace je
  „nic tu není" normální výsledek a nesmí vypadat jako selhání.

## Implementace

Nový modul **`ocr.py`**; `server.py` zůstane u routingu a HTTP.

`ocr.py` obsahuje:

- inicializaci RapidOCR nad lokálními modely,
- převod výstupu RapidOCR do tvaru odpovědi (polygon → bbox, filtrování, řazení, spojení textu),
- funkci `extract_text(image, min_confidence) -> dict`.

Inicializace (ověřeno proti `main.py` a `config.yaml` RapidOCR 3.9.2, `params` se mergují
do configu přes `ParseParams.update_batch`):

```python
RapidOCR(params={
    "Det.model_path": f"{MODELS_DIR}/PP-OCRv5_mobile_det.onnx",
    "Rec.model_path": f"{MODELS_DIR}/latin_PP-OCRv5_mobile_rec.onnx",
    "Rec.rec_keys_path": f"{MODELS_DIR}/latin_dict.txt",
    "Global.text_score": HARD_FLOOR,   # 0.1, viz níž
    "EngineConfig.onnxruntime.use_cuda": USE_CUDA,
})
```

**Kde se filtruje:** RapidOCR se inicializuje s pevným nízkým prahem `Global.text_score = 0.1`
a `min_confidence` z requestu se aplikuje až v našem kódu. Kdyby se práh z requestu předával
do RapidOCR, znamenalo by to přeinicializaci enginu na každý request a zároveň by nešlo
dostat bloky pod výchozí hodnotou. Práh 0.1 je jen odstranění zjevného šumu.

Konfigurace přes proměnné prostředí:

| Proměnná | Výchozí | Význam |
|---|---|---|
| `OCR_MODELS_DIR` | `./models` | Kde leží stažené ONNX a slovník |
| `OCR_USE_CUDA` | `auto` | `auto` = CUDA, je-li `CUDAExecutionProvider` mezi dostupnými, jinak CPU. `1` = vynutit CUDA (chybí-li, tvrdý pád při startu). `0` = vynutit CPU. |
| `OCR_MIN_CONFIDENCE` | `0.5` | Výchozí práh, když ho request neurčí |

Skripty:

- **`scripts/fetch_models.sh`** — stáhne obě ONNX z HuggingFace, vytáhne slovník
  z `inference.yml`, ověří SHA256 proti hodnotám zapsaným ve skriptu.
- **`scripts/sync-box.sh`** — rsync repa (bez `venv/`, `.git/`, `models/`) na box.

## Prostředí

**Vývoj a testy běží na boxu**, ne na Pi — Pi nemá GPU ani výkon na cokoli změřit.

- Git a psaní kódu zůstává na Pi; box nemá přístup na GitHub
  (`git@github.com: Permission denied (publickey)`, `gh` chybí), proto se synchronizuje rsyncem.
- Dev prostředí: `~/dev/image-embeddings` pod uživatelem `panbotka`, **vlastní venv**
  oddělený od produkčního `/opt/image-embeddings/venv`.
- **Dev server na portu 8010.** Port 8000 drží produkční `image-embeddings`,
  port 8001 `photo-enhancer`; ověřeno.
- Smyčka: úprava na Pi → `sync-box.sh` → testy a benchmark na boxu.

**Nasazení** do `/opt/image-embeddings` (vlastník `box`) a restart systemd služby
vyžadují sudo, které je k dispozici.

Instalační detail: `rapidocr` závisí na `opencv_python`, zatímco venv má
`opencv-python-headless<4.10` kvůli InsightFace. Obě distribuce obsazují stejný
`cv2` namespace. Řeší se instalací `rapidocr` s `--no-deps` a doinstalací zbytku
(`pyclipper`, `shapely`, `omegaconf`, `colorlog`, `PyYAML`, `six`, `tqdm`, `requests`)
s následným ověřením, že `import cv2` i face endpoint fungují dál.

Do produkce navíc přibude `onnxruntime-gpu` místo `onnxruntime`. InsightFace si dnes
v `server.py` vyžaduje `providers=["CPUExecutionProvider"]` explicitně, takže se jeho
chování nemá změnit — přesto to hlídá regresní kontrola níž.

## Chybové stavy

| Situace | Chování |
|---|---|
| Content-Type není `image/*` | 400, formulace konzistentní se stávajícími endpointy |
| Soubor Pillow neotevře | 400 s čitelnou zprávou, ne 500 |
| Fotka bez textu | 200, prázdný výsledek |
| Modely chybí nebo se nenačtou | Služba spadne **při startu**, ne až na prvním requestu |
| CUDA nedostupná | Fallback na CPU s WARNem, ne pád |

Velké fotky RapidOCR sám zmenšuje na `max_side_len: 2000` a bboxy mapuje zpět přes
`map_boxes_to_original`. Že to skutečně platí, ověřuje test, ne důvěra v dokumentaci.

## Testování

`tests/test_ocr.py` nad FastAPI `TestClient`:

1. Ne-image vstup → 400.
2. Bílý obrázek → 200, `text == ""`, `blocks == []`.
3. **Syntetický obrázek s českým textem** (Pillow + DejaVu vykreslí `PŘÍJEZD DO VESELICE 1978`)
   → OCR vrátí tentýž řetězec. Tohle je hlavní pojistka: chytí i situaci, kdy by se místo
   latinkového modelu načetl výchozí čínský.
4. Velká fotka (delší strana > 2000 px) → všechny bboxy leží uvnitř původních rozměrů.
5. `min_confidence` filtruje `text` i `blocks` shodně.

`scripts/bench_ocr.py` — propustnost (fotek/s) a výstupy k ručnímu posouzení, na CUDA i CPU.

- **Kvalita:** 7 fotek z kukátka dodaných zadavatelem (nápisy, cedule, SPZ). Vyhodnocuje se
  ručně — u fotek z archivu není k dispozici referenční přepis.
- **Propustnost:** opakované běhy nad těmi samými fotkami a nad syntetickou sadou obrázků
  v několika rozlišeních (1 Mpx až 12 Mpx), aby měření nezáviselo na dalších datech.
  Měří se medián a 95. percentil doby na fotku, zvlášť pro CUDA a pro CPU.

**Regresní kontrola po instalaci `onnxruntime-gpu`:** `/embed/image`, `/embed/text`,
`/embed/face` a `/estimate/era` musí vracet totéž co před ní.

## Akceptační kritéria

- `pytest` prochází na boxu.
- Syntetický test přečte českou diakritiku správně.
- Všech 7 dodaných fotek vrátí smysluplný text (ruční posouzení).
- Propustnost změřena na CUDA i CPU a zapsána.
- `/health` ukazuje provider a název OCR modelu.
- Stávající čtyři endpointy po zásahu do venvu fungují beze změny.

## Rizika

| Riziko | Ošetření |
|---|---|
| `onnxruntime-gpu` rozbije InsightFace | Explicitní `CPUExecutionProvider` už v kódu je; regresní kontrola všech endpointů |
| Kolize `opencv_python` × `opencv-python-headless` | Instalace `--no-deps`, ověření `import cv2` a face endpointu |
| PP-OCRv6 by mohl být lepší | RapidOCR má `multi_PP-OCRv6_rec_small` jako nový výchozí, jazykové pokrytí ale doložené není. Porovná se v benchmarku a přepne se, jen když na češtině vyhraje. |
| VRAM (volné ~2,3 GB z 8 GB) | Mobile modely jsou jednotky MB; spotřebu ověří benchmark |
