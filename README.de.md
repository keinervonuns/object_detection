> 🌐 **Deutsch** · [English](README.md)

# PI Detection

Multi-Klassen-Bildklassifikation mit Echtzeit-Kamera-Inferenz. Trainiere auf einer
**NVIDIA RTX A2000** (oder einer beliebigen CUDA-GPU, oder CPU) und deploye auf
einen **Raspberry Pi 5**, der den Live-Feed ausführt und ein **4-Bit-GPIO-LED**-
Display ansteuert.

Der Workflow: Bilder sammeln → train/val aufteilen → trainieren → Modell auf den
Pi kopieren → Erkennung starten, mit GradCAM + OmniXAI (LIME)-Erklärbarkeit und
Region-Maskierung.

Alles liegt unter [`Torch/`](Torch/).

---

## Inhalt

- [Projektstruktur](#projektstruktur)
- [1. Welches Setup?](#1-welches-setup)
- [2. Umgebung einrichten](#2-umgebung-einrichten)
- [3. Klassen konfigurieren](#3-klassen-konfigurieren)
- [4. Bilder sammeln](#4-bilder-sammeln)
- [5. Aufteilen in Train und Val](#5-aufteilen-in-train-und-val)
- [6. Trainieren](#6-trainieren)
- [7. Modell auf den Pi kopieren](#7-modell-auf-den-pi-kopieren)
- [8. Erkennung starten](#8-erkennung-starten)
- [Erkennungsbereich](#erkennungsbereich)
- [GPIO-LED-Ausgabe](#gpio-led-ausgabe)
- [Tests](#tests)
- [Gespeicherte Dateien](#gespeicherte-dateien)
- [Kamera wechseln](#kamera-wechseln)
- [Backbone wechseln](#backbone-wechseln)
- [Genauigkeit verbessern](#genauigkeit-verbessern)
- [Fehlerbehebung](#fehlerbehebung)

---

## Projektstruktur

```
Torch/
├── source/                  ← HIER KOMMEN DEINE ROHBILDER HIN (ein Ordner pro Klasse)
│   ├── background/
│   ├── blue_cube/
│   ├── gray_cuboid/
│   ├── green_cube/
│   ├── pink_cylinder/
│   ├── silver_cube/
│   └── silver_cylinder/
│
├── data/                    ← automatisch erzeugt von setup_data.py (nicht bearbeiten)
│   ├── train/
│   └── val/
│
├── models/                  ← automatisch erzeugt von train.py
│   ├── best_model.pth       ← die einzige Datei, die der Pi für die Inferenz braucht
│   ├── training_curves.png
│   ├── classification_report.txt
│   └── training_history.json
│
├── captures/                ← geschrieben, wenn du im Live-Feed `s` drückst
│
├── regions/
│   └── region_config.ini    ← gespeichertes Erkennungsrechteck (e / Pfeiltasten)
│
├── tests/                   ← eigenständige Hardware-Checks
│   ├── test_camera.py       ← Kamera-Check (Index / Auflösung)
│   └── test_led.py          ← LED-Verkabelungstest (--mode single | --mode binary)
│
├── config.py                ← alle Einstellungen (hier anfangen)
├── setup_data.py            ← erstellt Ordner + teilt Bilder auf
├── train.py                 ← trainiert das Modell
├── capture_images.py        ← Webcam-Werkzeug zum Sammeln von Trainingsbildern
├── run.py                   ← ★ Einstiegspunkt: Echtzeit-Inferenz + GradCAM + GPIO-LEDs
└── requirements.txt
```

`source/`, `data/`, `models/`, `captures/` und `regions/` werden zur Laufzeit
erstellt/genutzt; `data/*` und `models/*` sind git-ignoriert.

---

## 1. Welches Setup?

Du kannst auf einem GPU-Rechner trainieren und das Modell auf den Pi kopieren —
oder direkt auf dem Pi trainieren (nur CPU, deutlich langsamer). Beides nutzt
denselben Code und dasselbe `Torch/`-Verzeichnis.

| | Option A — Training auf einem GPU-Rechner *(empfohlen)* | Option B — alles auf dem Pi |
|---|---|---|
| Training | Windows-/Linux-Rechner mit CUDA-GPU (z. B. RTX A2000) | Raspberry Pi 5, **nur CPU** |
| Geschwindigkeit | Minuten pro Epoche | Viele Minuten pro Epoche |
| Modell-Transfer | `scp`/`rsync` von `models/best_model.pth` auf den Pi | — |
| Inferenz | Pi (GPIO-LEDs) | Pi (GPIO-LEDs) |

Richte die Umgebungen ein, die du brauchst — siehe
[Abschnitt 2](#2-umgebung-einrichten).

---

## 2. Umgebung einrichten

Überall Python 3.10+ (entwickelt auf 3.10–3.13; der Pi 5 läuft mit 3.13).

### 2.1 GPU-Trainingsrechner — Windows

PyTorch liefert unter Windows bei aktuellen Versionen standardmäßig ein
CUDA-fähiges Wheel.

```powershell
cd Torch

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
venv\Scripts\Activate.ps1
#   Falls PowerShell das Skript blockiert, für diese Sitzung erlauben:
#   Set-ExecutionPolicy -Scope Process Bypass

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional — OmniXAI-Erklärungsberichte (die `s`-Taste schreibt eine HTML-Datei).
# Nutze die schmalen Extras und wende die Pins danach erneut an (siehe Hinweis unten).
pip install "omnixai[vision,plot]"
pip install -r requirements.txt
```

Prüfe, ob die GPU erkannt wird:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Hinweise zu Windows:
- Falls der `DataLoader` mit Worker-/`spawn`-Problemen abbricht, setze
  `NUM_WORKERS = 0` in **config.py**.
- Symlinks benötigen den **Entwicklermodus** (siehe
  [Schritt 5](#5-aufteilen-in-train-und-val)); setze stattdessen
  `COPY_FILES = True`, wenn du das überspringen willst.

### 2.2 GPU-Trainingsrechner — Linux

```bash
cd Torch
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "omnixai[vision,plot]"   # optional — siehe OmniXAI-Hinweis unten
```

**GPU:** Falls `torch.cuda.is_available()` `False` ergibt, installiere das zu
deinem Treiber passende CUDA-Wheel von
<https://pytorch.org/get-started/locally/>. Das CUDA-Toolkit / der NVIDIA-Treiber
selbst muss separat installiert werden
(<https://developer.nvidia.com/cuda-downloads>).

### 2.3 Raspberry Pi 5

Kopiere das gesamte Projekt auf den Pi. GPIO (`RPi.GPIO`) liegt im
**System**-Python, während die schweren Pakete aus der venv kommen.

```bash
cd ~/pidetection/Torch     # dorthin, wo du das Projekt legst

# venv mit demselben Python wie das System-python3 erstellen (3.13 auf dem Pi 5)
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install RPi.GPIO
deactivate
```

> `run.py` fügt beim Start `venv/lib/python*/site-packages` zum Pfad des
> **System**-Interpreters hinzu, sodass `sudo python3 run.py` `RPi.GPIO` vom
> System und `torch` / `cv2` aus der venv bezieht.

`omnixai` ist überall optional: Es wird verzögert und abgesichert importiert.
Fehlt es, läuft der Live-Feed trotzdem mit Echtzeit-GradCAM und der vollständigen
Oberfläche — du verlierst nur den HTML-Bericht, den die `s`-Taste schreibt. Der
Rest von `requirements.txt` (torch, torchvision, opencv, numpy) ist erforderlich.

> **`omnixai` installieren:** Nutze `pip install "omnixai[vision,plot]"`, **nicht**
> `omnixai[all]`. `[all]` zieht zusätzlich die NLP-/BentoML-Stacks, und die
> vision-Extras installieren **`opencv-python-headless`**, das keine GUI-Funktionen
> hat — `cv2.imshow` / `cv2.namedWindow` würden fehlschlagen. Führe danach erneut
> `pip install -r requirements.txt` aus, damit die gepinnten `opencv-python` und
> `scikit-learn>=1.3` gewinnen (das Release von omnixai aus 2023 kann sie sonst
> herunterstufen).

---

## 3. Klassen konfigurieren

Öffne **config.py** und passe das `CLASSES`-Dict an deine Objektnamen an:

```python
CLASSES = {
    "background": 0,
    "green_cube": 1,
    "blue_cube": 2,
    "pink_cylinder": 3,
    "silver_cylinder": 4,
    "silver_cube": 5,
    "gray_cuboid": 6
}
```

**Dieses Dict definiert deine Klassen — du erstellst die Ordner nicht von Hand.**
Die Schlüssel müssen exakt den Unterordnernamen unter `source/` entsprechen, und
Schritt 4 erstellt diese Ordner *aus diesem Dict*. Halte `CLASSES` auf dem
Trainingsrechner und dem Pi identisch.

Die **Werte sind die Zahl, die bei Vorhersage dieser Klasse an die GPIO-LEDs
gesendet wird** (siehe [GPIO-LED-Ausgabe](#gpio-led-ausgabe)). Halte sie
eindeutig (zwei Klassen mit derselben Zahl sehen auf den LEDs identisch aus) und
im Bereich `0–15` — sie werden nicht validiert, daher erzeugt ein Wert über `15`
einen 5-Bit-String, läuft über die 4-Pin-Liste hinaus und leuchtet still das
falsche Muster. Halte die Zahlen stabil, sobald deine Verkabelung steht.

> **Die Ordnererkennung wird nicht von diesem Dict gesteuert.** `setup_data.py`
> teilt alle Ordner auf, die es in `source/` findet — auch solche, die in
> `CLASSES` fehlen. `train.py` gibt einen NOTE aus, wenn die gefundenen Ordner
> von `config.CLASSES` abweichen — da es die *alphabetisch sortierte*
> Ordnerliste mit der *Einfügereihenfolge* des Dicts vergleicht, erscheint dieser
> NOTE in diesem Projekt selbst dann, wenn die Klassen korrekt sind (die
> Reihenfolge weicht ab, die Menge nicht). `run.py` schlägt Klassen mit
> `numbers.get(class_name, 0)` über den Namen nach, daher gibt eine Klasse, die
> im Modell, aber nicht in `CLASSES` vorhanden ist, `0` auf den LEDs aus.

---

## 4. Bilder sammeln

Erstelle das Ordnergerüst aus `CLASSES`:

```bash
python setup_data.py --init
```

Lege dann Bilder in jeden `source/<class>/`-Ordner:

- Unterstützte Formate: `.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`
- Strebe **mindestens 50–200 Bilder pro Klasse** an
- Mehr Daten schlagen ein größeres Modell; variiere Winkel, Abstände, Licht und Hintergründe

> **Abkürzung:** `python capture_images.py` öffnet die Webcam und speichert
> Frames direkt in `source/<class>/` als `00000.jpg`, `00001.jpg`, …  
> Steuerung: `SPACE` speichern · `c` Auto-Speichern alle 500 ms · `n` / `p`
> nächste / vorherige Klasse · `q` beenden. Die Klassen wechseln in der
> Reihenfolge von `config.CLASSES`. Benötigt ein Display — führe es auf dem
> Rechner mit der Webcam aus.

---

## 5. Aufteilen in Train und Val

```bash
python setup_data.py
```

Mischt und teilt deine Quellbilder auf (löscht und baut `data/` bei jedem Lauf
neu auf):

| Split | Standard | Gesteuert durch |
|-------|----------|-----------------|
| Train | 80 %     | `VAL_SPLIT` |
| Val   | 20 %     | `VAL_SPLIT` |

| Einstellung | Standard | Beschreibung |
|-------------|----------|--------------|
| `VAL_SPLIT` | `0.2` | Anteil, der für die Validierung zurückgehalten wird |
| `SPLIT_SEED` | `42` | Zufalls-Seed — für einen anderen Split ändern |
| `COPY_FILES` | `False` | `False` = Symlinks (spart Speicher); `True` = Kopien |

> Unter Windows benötigen Symlinks den **Entwicklermodus** (`Einstellungen →
> System → Für Entwickler → Entwicklermodus`); schlägt es fehl, fällt das Skript
> auf Kopieren zurück. Auf dem Pi bei `COPY_FILES = False` bleiben — Symlinks
> sparen Platz auf der 32-GB-Karte.

---

## 6. Trainieren

### 6.1 Auf einem GPU-Rechner (empfohlen)

Mit aktivierter venv, aus `Torch/`:

```bash
# Windows: venv\Scripts\Activate.ps1     Linux: source venv/bin/activate
python train.py
```

1. Lädt `data/train` und `data/val`
2. Fine-tuned das ImageNet-vortrainierte `BACKBONE` auf deine Klassen
3. Augmentiert jede Epoche: Crop, Flip, Rotation, Colour Jitter, Random Erase, MixUp
4. Nutzt **Mixed Precision (AMP)** — schneller und weniger VRAM auf der A2000
5. Speichert den besten Checkpoint nach `models/best_model.pth`
6. Bricht früh ab nach `EARLY_STOP` Epochen ohne Verbesserung
7. Schreibt `models/training_curves.png`, `models/classification_report.txt`,
   `models/training_history.json`

| Einstellung | Standard (A2000) | Hinweise |
|-------------|------------------|----------|
| `BACKBONE` | `resnet50` | Auch `efficientnet_b0`, `efficientnet_b2` |
| `IMG_SIZE` | `224` | Native Größe für alle Backbones |
| `BATCH_SIZE` | `32` | Bei CUDA Out-of-Memory auf `16` senken |
| `NUM_EPOCHS` | `50` | Early Stopping kann früher beenden |
| `LEARNING_RATE` | `3e-4` | AdamW-Lernrate |
| `WEIGHT_DECAY` | `1e-4` | AdamW Weight Decay |
| `SCHEDULER` | `cosine` | Oder `step` (`STEP_LR_DECAY` / `STEP_LR_STEP`) |
| `EARLY_STOP` | `8` | Epochen ohne Verbesserung vor dem Abbruch |
| `LABEL_SMOOTHING` | `0.1` | Hilft bei kleinen Datensätzen |
| `USE_AMP` | `True` | Für die A2000 aktiviert lassen |
| `MIXUP_ALPHA` | `0.2` | MixUp-Stärke; `0` deaktiviert |
| `NUM_WORKERS` | `4` | Auf `0` setzen, falls der DataLoader Fehler wirft (Windows) |

### 6.2 Auf dem Raspberry Pi (nur CPU)

Der Pi 5 hat keine CUDA-GPU, das Training läuft also auf der CPU — es
funktioniert, aber rechne mit Minuten pro Epoche. Nutze das kleinste Backbone und
eine kleine Batch.

Passe **config.py** auf dem Pi an:

```python
BACKBONE    = "efficientnet_b0"   # schnellstes Backbone
BATCH_SIZE  = 8                   # 4, falls der RAM knapp wird (8 GB gesamt)
NUM_WORKERS = 2
USE_AMP     = False               # AMP ist CUDA-only (auf CPU ohnehin ignoriert)
PIN_MEMORY  = False               # kein GPU-Pinning nötig
```

Dann, mit aktivierter venv:

```bash
source venv/bin/activate
python train.py
```

Hinweise zum Training auf dem Pi:
- `train.py` öffnet am Ende ein matplotlib-Fenster. Auf einem headless Pi wird
  nur eine Warnung ausgegeben — der Plot wird trotzdem nach
  `models/training_curves.png` gespeichert. Führe es vom Desktop des Pi oder per
  `ssh -X` aus, wenn du ihn sehen willst.
- Halte den Datensatz klein (einige Hundert Bilder), damit ein Lauf in
  vertretbarer Zeit endet; andernfalls auf einem GPU-Rechner trainieren und
  [das Modell kopieren](#7-modell-auf-den-pi-kopieren).
- Das Ergebnis ist dieselbe `models/best_model.pth` wie auf einem GPU-Rechner —
  genau die Datei, die der Live-Feed lädt.

---

## 7. Modell auf den Pi kopieren

Für die Inferenz wird nur **`models/best_model.pth`** gebraucht: Die Datei
enthält die Klassenliste, den Backbone-Namen und die Gewichte und ist damit
in sich geschlossen. Kopiere genau diese eine Datei vom Trainingsrechner auf den
Pi.

Lege den Zielordner auf dem Pi vorher an (er ist git-ignoriert und evtl. nicht
vorhanden):

```bash
# auf dem Pi
mkdir -p ~/pidetection/Torch/models
```

**Von Linux/macOS (Trainingsrechner):**

```bash
# scp
scp Torch/models/best_model.pth pi@raspberrypi.local:/home/pi/pidetection/Torch/models/

# oder rsync (fortsetzbar, mit Fortschritt)
rsync -avP Torch/models/best_model.pth pi@raspberrypi.local:/home/pi/pidetection/Torch/models/
```

**Von Windows (PowerShell, integriertes OpenSSH-`scp`):**

```powershell
scp .\Torch\models\best_model.pth pi@raspberrypi.local:/home/pi/pidetection/Torch/models/
```

Ersetze `pi@raspberrypi.local` durch Benutzer und Host (oder IP) deines Pi.
Alternativen: die Datei auf einen USB-Stick kopieren oder den Modellordner über
das Netzwerk freigeben (z. B. Samba).

> Trainiere auf dem GPU-Rechner neu und wiederhole dieses Kopieren, um das Modell
> auf dem Pi zu aktualisieren. Du kannst das Gesamtprojekt auch einmal kopieren
> und danach nur noch `models/best_model.pth` übertragen — Code und
> `requirements.txt` ändern sich selten.

---

## 8. Erkennung starten

`run.py` ist der einzige Einstiegspunkt. Die GPIO-LED-Ausgabe ist
**standardmäßig an**; übergib `--no-gpio`, um sie zu deaktivieren.

### 8.1 Alles auf dem Pi (mit GPIO-LEDs)

Der Pi benötigt das trainierte Modell aus
[Schritt 7](#7-modell-auf-den-pi-kopieren), eine angeschlossene Kamera und die
verkabelten LEDs (siehe [GPIO-LED-Ausgabe](#gpio-led-ausgabe)).

```bash
sudo python3 run.py
```

Beim Start erledigt `run.py`:

1. parst `--no-gpio` und initialisiert, wenn GPIO aktiviert ist, `RPi.GPIO`
   **bevor** `torch` / `cv2` importiert werden (ein Import davor kann das
   GPIO-Setup stören),
2. fügt die `site-packages` der Projekt-venv zu `sys.path` hinzu (sodass
   `torch` / `cv2` aus der venv kommen, während `RPi.GPIO` vom System kommt),
3. startet den Feed.

Für den GPIO-Zugriff ist `sudo` erforderlich. Schlägt die GPIO-Initialisierung
fehl, warnt das Skript und läuft mit deaktivierten LEDs weiter (übergib
`--no-gpio`, um die Warnung zu überspringen). Alle Pfade sind absolut (aufgelöst
über `config.BASE_DIR`), sodass das Projekt überall liegen und aus jedem
Verzeichnis gestartet werden kann.

Der Betrieb unter `sudo` benötigt X-Zugriff — starte vom Desktop des Pi oder
erlaube es einmalig mit `xhost +local:root` (oder starte den Prozess mit
`sudo -E`).

### 8.2 Auf dem Trainingsrechner (ohne GPIO)

```bash
# Windows: venv\Scripts\Activate.ps1     Linux: source venv/bin/activate
python run.py --no-gpio
```

Praktisch, um das Modell auf dem Rechner zu testen, der es trainiert hat — ohne
Pi.

In beiden Fällen wird die Kamera als **Index 0, V4L2, MJPG, 1920×1080, Buffer 1**
in einem Vollbildfenster geöffnet. Ein zweiter Capture-Handle (Index 1) wird beim
Start kurz geöffnet, nur um die Frame-Größe für den Region-Standard zu lesen.

**Tastatursteuerung:**

| Taste | Aktion |
|-------|--------|
| `G` | GradCAM-Heatmap-Overlay umschalten |
| `S` | Aktuellen Frame + OmniXAI-HTML-Bericht speichern |
| `E` | Region-Bearbeitungsmodus betreten / verlassen (speichert das Rechteck beim Verlassen) |
| `R` | FPS-Zähler zurücksetzen |
| `H` | Hilfe im Terminal ausgeben |
| `Q` | Beenden — oder im Region-Bearbeitungsmodus die Region speichern und den Modus verlassen |

**Im Region-Bearbeitungsmodus:**

| Taste | Aktion |
|-------|--------|
| Pfeiltasten | Aktive Kante verschieben (`SHRINK` bewegt sie nach innen, `EXPAND` nach außen) |
| `S` | Expand / Shrink umschalten (außerhalb des Modus *speichert* `s`) |
| `Z` | Letzte Kantenbewegung rückgängig machen |
| `C` | Region auf das volle Bild zurücksetzen |
| `E` / `Q` | Bestätigen, nach `regions/region_config.ini` speichern, Modus verlassen |

**Bildschirmanzeige:** oberes Banner (Klasse + Konfidenzbalken), Seitenleiste
(Top-3-Wahrscheinlichkeiten), untere Leiste (FPS, GradCAM-Status, Tastenhinweise)
und eine abgedunkelte Umrandung, die den Bereich außerhalb der Erkennungsregion
zeigt (gelb beim Bearbeiten, sonst grün).

> Das Klassen-Label ändert sich erst, nachdem es `DISPLAY_SECONDS` (Standard
> `1.0`) stabil war, um Flackern zu vermeiden; die Konfidenz wird derweil weiter
> aktualisiert.

---

## Erkennungsbereich

Nur das Rechteck in `regions/region_config.ini` wird dem Modell zugeführt; alles
außerhalb wird vor der Inferenz auf 0 gesetzt und auf dem Bildschirm abgedunkelt.

```ini
[region]
left = 1056
top = 504
right = 1696
bottom = 744
```

Drücke `e`, verschiebe die Kanten mit den Pfeiltasten und drücke erneut `e` zum
Speichern. Das Rechteck wird am Frame begrenzt und kann nicht unter 40 px pro
Seite schrumpfen.

---

## GPIO-LED-Ausgabe

Die vorhergesagte Klasse wird als **4-Bit-Binärzahl** auf vier GPIO-Pins
ausgegeben, einmal pro Inferenz im Hintergrund-Thread aktualisiert.

| Pin (BCM) | Bit | Wert |
|-----------|-----|------|
| 17 | Bit 0 | 1 (LSB) |
| 27 | Bit 1 | 2 |
| 22 | Bit 2 | 4 |
| 23 | Bit 3 | 8 (MSB) |

Die Zahl stammt aus dem `CLASSES`-Dict, z. B. leuchtet bei `blue_cube → 2` nur
Pin 27.

**Verkabelung (active-LOW):** 3,3 V → LED-Anode; LED-Kathode → ~220 Ω Widerstand
→ GPIO-Pin. Der Pin zieht den Strom, daher wird ein Bit auf `LOW` gezogen, um es
einzuschalten.

**Die Pins werden active-LOW angesteuert** — ein Bit ist `LOW` für `1` und
`HIGH` für `0`, und „aus" setzt alle vier Pins auf `HIGH`. `tests/test_led.py`
nutzt in beiden Modi dieselbe Konvention, sodass ein Verkabelungstest damit zum
Live-Feed passt. Prüfe mit `tests/test_led.py`, bevor du die vollständige
Pipeline startest.

---

## Tests

Eigenständige Hardware-Checks liegen in `tests/` (führe sie aus `Torch/` aus):

```bash
python tests/test_camera.py            # Live-Vorschau + tatsächliche Auflösung
python tests/test_led.py               # einzelne LED an GPIO 17
python tests/test_led.py --mode binary # 4-LED-Binäranzeige (0–15)
```

- `test_camera.py` — bestätigt Kamera-Index und Auflösung; `q` beendet.
- `test_led.py --mode single` — lässt eine LED an GPIO 17 blinken (gpiozero → Mock
  → RPi.GPIO als Fallbacks).
- `test_led.py --mode binary` — zeigt `0–15` auf dem LED-Array an; gib einen Wert
  ein, `16` zum Durchlaufen aller Werte, `q` zum Beenden. GPIO-Zugriff benötigt
  normalerweise `sudo`.

---

## Gespeicherte Dateien

`S` **außerhalb des Region-Bearbeitungsmodus** schreibt nach `captures/`:

- `captures/capture_NNNN_<class>.jpg` — annotierter Frame
- `captures/explanation_NNNN_<class>.html` — OmniXAI-GradCAM-+-LIME-Bericht

> Der `NNNN`-Zähler wird bei jedem Speichern erhöht, sodass aufeinanderfolgende
> Speicherungen einander nicht überschreiben.

---

## Kamera wechseln

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

Ändere das erste Argument (`0` → `1`, `2`, …) für ein anderes Gerät. `CAP_V4L2`
ist nur unter Linux verfügbar — entferne es unter Windows/macOS. `run.py`,
`capture_images.py` und `tests/test_camera.py` teilen diese Einstellungen, also
passe alle drei gemeinsam an.

Es gibt außerdem einen **zweiten** Capture bei Index `1`, der nur geöffnet wird,
um die als Region-Standard genutzten Frame-Maße zu lesen; schlägt er fehl, fällt
er still auf `1920×1080` zurück. Führe `python tests/test_camera.py` aus, um Index
und Auflösung zu bestätigen.

> **Auf dem Pi:** Der Feed öffnet ein **V4L2**-Gerät, daher funktioniert eine
> **USB-Webcam** sofort. Für die CSI-Flachkamera die V4L2-Kompatibilitätsschicht
> aktivieren (`libcamera` → `bcm2835-v4l2`), damit sie als `/dev/video*`
> erscheint.

---

## Backbone wechseln

Ändere `BACKBONE` in **config.py** und trainiere neu:

| Backbone | VRAM | Geschwindigkeit | Genauigkeit |
|----------|------|-----------------|-------------|
| `efficientnet_b0` | ~2 GB | Am schnellsten | Gut |
| `efficientnet_b2` | ~3 GB | Mittel | Besser |
| `resnet50` | ~4 GB | Mittel | Vergleichbar |

`efficientnet_b0` ist die Wahl für das Training auf dem Pi (CPU); `resnet50` /
`efficientnet_b2` sind für den GPU-Rechner. `run.py` baut die Architektur aus dem
im Checkpoint gespeicherten `backbone`-Feld neu auf, daher ist dort keine
Änderung nötig.

---

## Genauigkeit verbessern

- **Variiere deine Quellbilder** — Winkel, Abstände, Lichtverhältnisse, Hintergründe
- **Balanciere deine Klassen** — ähnliche Bildanzahl pro Klasse
- **Nimm eine `background`-Klasse auf** — Szenen ohne Zielobjekt reduzieren Fehlalarme
- **Mehr Daten schlagen ein größeres Modell** — verdopple deine Bilder, bevor du das Backbone wechselst
- **Prüfe den Classification Report** — geringer Recall pro Klasse bedeutet, dass diese Klasse mehr Bilder braucht

---

## Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| `CUDA out of memory` | `BATCH_SIZE` auf `16` oder `8` senken |
| `torch.cuda.is_available()` ist `False` | Installiere das zu deinem Treiber passende CUDA-Wheel von pytorch.org |
| `Import torch could not be resolved` (VS Code) | Wähle den venv-Interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → `Torch/venv` |
| PowerShell führt `Activate.ps1` nicht aus | `Set-ExecutionPolicy -Scope Process Bypass`, dann erneut ausführen |
| `Cannot open camera` | Index/Auflösung mit `python tests/test_camera.py` prüfen; die Kamera nutzt `CAP_V4L2` (nur Linux) |
| `No trained model found` | Erst trainieren (`python train.py`) oder [das Modell auf den Pi kopieren](#7-modell-auf-den-pi-kopieren) |
| Symlink-Fehler unter Windows | Entwicklermodus aktivieren oder `COPY_FILES = True` setzen |
| Sehr geringe Genauigkeit | Mehr Bilder hinzufügen; prüfen, ob die Ordnernamen exakt `CLASSES` entsprechen |
| `NUM_WORKERS`-Fehler unter Windows | `NUM_WORKERS = 0` in config.py setzen |
| Training auf dem Pi ist extrem langsam | Erwartet — `BACKBONE = "efficientnet_b0"`, `BATCH_SIZE = 8` nutzen oder auf einem GPU-Rechner trainieren |
| `⚠ GPIO unavailable … LED output disabled` | Mit `sudo python3 run.py` auf dem Pi ausführen oder `--no-gpio` auf einem Rechner ohne GPIO übergeben |
| Matplotlib-Fenster erscheint auf dem Pi nicht | Headless-Pi — der Plot wird trotzdem nach `models/training_curves.png` gespeichert |
| LEDs leuchten invertiert | Die Verkabelung ist active-HIGH — der Code steuert active-LOW (Bit = Pin `LOW`); siehe [GPIO-LED-Ausgabe](#gpio-led-ausgabe) |
| `cannot open display` unter sudo | X-Zugriff erlauben: `xhost +local:root` (oder mit `sudo -E` ausführen) |
| `omnixai`-Explainer nicht verfügbar | Nur für `s` nötig; GradCAM funktioniert weiterhin |
| Region ohne Wirkung | INI fehlt → drücke `e`, um eine zu zeichnen und zu speichern |
| Erkennt Objekte außerhalb der Region | Drücke `e`, dann `c` zum Zurücksetzen, oder speichere ein breiteres Rechteck |
