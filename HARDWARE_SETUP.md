# Hardware Setup Guide
## Edge-AI Hypoxia Early Warning System

---

## 1. Arduino UNO ↔ Raspberry Pi 5 — USB Cable

> **Why USB instead of GPIO UART?**  
> Connecting via a standard **USB-A to USB-B cable** is simpler, safer, and
> requires **no voltage-divider resistors**. The USB interface on the Arduino
> UNO handles the 5 V ↔ 3.3 V level shifting internally.
> 
> The Arduino's `Serial` object already maps to the USB port, so **no Arduino
> code changes are needed**. The Raspberry Pi sees the Arduino as a standard
> USB CDC serial device (`/dev/ttyACM0` for genuine UNOs or `/dev/ttyUSB0`
> for CH340-based clones).

---

### Wiring Diagram

```
 ARDUINO UNO                       RASPBERRY PI 5
 ─────────────                     ─────────────────────

  USB-B port ──── USB-A to USB-B cable ──── USB-A port
       │                                         │
  (power + data)                          (power + data)

  No GPIO wires.  No resistors.  No voltage divider.
```

### Required Cable

| Cable type | Notes |
|---|---|
| USB-A (RPi side) to USB-B (Arduino side) | Standard "printer" cable — ships with most Arduino kits |

> ⚠️ Make sure to use a **data cable**, not a charge-only cable. Charge-only
> cables have no D+/D− lines and will not enumerate as a serial port.

---

## 2. MAX30102 Wiring (Arduino UNO)

| MAX30102 Pin | Arduino Pin | Notes |
|---|---|---|
| VIN | 3.3 V | Use Arduino's 3.3 V reg output |
| GND | GND | |
| SDA | A4 | I²C data |
| SCL | A5 | I²C clock |
| INT | Not connected | Optional interrupt pin |

> The SparkFun MAX30102 breakout includes on-board 4.7 kΩ pull-ups on SDA/SCL.

---

## 3. DS18B20 Wiring (Arduino UNO)

```
DS18B20            Arduino UNO
────────           ───────────
VDD ───────────── 5 V
GND ───────────── GND
DATA ──┬────────── D4
       │
      4.7 kΩ
       │
       └────────── 5 V   ← pull-up resistor (mandatory)
```

> Use a **waterproof DS18B20 probe** for body temperature sensing.  
> Place it in contact with the fingertip or wrist alongside the MAX30102.

---

## 4. LED & Buzzer Wiring (Arduino UNO)

| Component | Arduino Pin | Resistor | Notes |
|---|---|---|---|
| Green LED (+) | D5 | 220 Ω → GND | Normal state |
| Yellow LED (+) | D6 | 220 Ω → GND | Warning (PWM pulsing) |
| Red LED (+) | D7 | 220 Ω → GND | Critical state |
| Active Buzzer (+) | D8 | Direct | Active type (built-in oscillator) |
| All LED (−) | GND | — | |
| Buzzer (−) | GND | — | |

---

## 5. Raspberry Pi 5 — USB Serial Setup

### Step 1: Add your user to the dialout group
```bash
sudo usermod -aG dialout $USER
# Log out and back in for the group change to take effect
```

### Step 2: Verify the USB device appears
```bash
# Plug the USB cable into the Arduino and the Raspberry Pi, then:
ls /dev/ttyACM* /dev/ttyUSB*
# Expected output (genuine UNO):
#   /dev/ttyACM0
# Expected output (CH340 clone):
#   /dev/ttyUSB0
```

### Step 3: Test the USB connection
```bash
# With Arduino running its sketch:
cat /dev/ttyACM0
# You should see lines like:  98,72,36.85,1.234
# Press Ctrl-C to stop.
```

> **Tip:** The Python script (`rpi5_inference.py`) auto-detects the port
> by trying `/dev/ttyACM0` then `/dev/ttyUSB0`, so no manual configuration
> is required.

---

## 6. Raspberry Pi 5 — Python Environment Setup

```bash
# Create a virtual environment (required on RPi OS Bookworm+)
python3 -m venv ~/hypoxia_env
source ~/hypoxia_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pyserial numpy requests

# Install tflite-runtime (ARM64 wheel for RPi 5)
pip install tflite-runtime
# If the above fails, use the Coral wheel:
# pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
```

---

## 7. Deploy Model Files to Raspberry Pi

After running **Script 1** on your PC/Colab, copy these two files to the Pi:

```bash
# From PC (Windows PowerShell):
scp hypoxia_hybrid_model.tflite  pi@<raspberry_pi_ip>:~/
scp scaler_params.json           pi@<raspberry_pi_ip>:~/

# On Raspberry Pi, run inference:
cd ~
source hypoxia_env/bin/activate
python3 rpi5_inference.py
```

---

## 8. Run at Boot (systemd service)

```bash
sudo nano /etc/systemd/system/hypoxia.service
```

Paste:
```ini
[Unit]
Description=Hypoxia Early Warning Edge AI
After=network.target

[Service]
ExecStart=/home/pi/hypoxia_env/bin/python3 /home/pi/rpi5_inference.py
WorkingDirectory=/home/pi
Restart=always
RestartSec=5
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable hypoxia
sudo systemctl start hypoxia
sudo journalctl -fu hypoxia   # live logs
```

---

## 9. Complete System Block Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Patient / User                        │
│           MAX30102 (finger clip) + DS18B20 (probe)      │
└──────────────────┬──────────────────────────────────────┘
                   │ I²C / 1-Wire
┌──────────────────▼──────────────────────────────────────┐
│                ARDUINO UNO (5 V)                         │
│  Moving Average Filter → SpO2, HR, PI, Temp             │
│  USB port ←──── USB-A to USB-B cable ────► USB port     │
│  (Serial @ 115200 baud, CDC/ACM via USB)                │
│  Actuators: Green/Yellow/Red LED + Buzzer               │
└──────────────────┬──────────────────────────────────────┘
                   │ USB 115200 baud (/dev/ttyACM0)
┌──────────────────▼──────────────────────────────────────┐
│              RASPBERRY PI 5                              │
│  1. pyserial  → parse "SpO2,HR,Temp,PI"                 │
│  2. Kalman Filter (×4) → remove motion artifacts        │
│  3. Sliding window buffer (30 samples)                  │
│  4. TFLite inference → Normal / Warning / Critical      │
│  5. Send 'N'/'W'/'C' back to Arduino via USB            │
│  6. Log to vitals.csv                                   │
│  7. Push to ThingSpeak every 15 s (with retry)         │
└──────────────────┬──────────────────────────────────────┘
                   │ Wi-Fi / Ethernet
┌──────────────────▼──────────────────────────────────────┐
│          ThingSpeak Dashboard (Channel 3375745)          │
│  Field1=SpO2  Field2=HR  Field3=Temp  Field4=PI         │
│  Field5=Prediction  Field6=Confidence%                  │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Arduino Library Installation

Open **Arduino IDE 2.x** → Tools → Manage Libraries, then search and install:

| Library | Version tested | Notes |
|---|---|---|
| SparkFun MAX3010x Pulse and Proximity Sensor | ≥ 1.1.1 | Includes `spo2_algorithm.h` |
| OneWire | ≥ 2.3.7 | By Paul Stoffregen |
| DallasTemperature | ≥ 3.9.0 | By Miles Burton |

---

## 11. ThingSpeak Dashboard Setup

1. Go to [thingspeak.com](https://thingspeak.com) → Your channel **3375745**
2. Add widgets: **Numeric display** for SpO2, HR, Temp; **Gauge** for PI
3. Add a **Line chart** spanning all 6 fields for trend monitoring
4. Set **channel update interval** to 15 s (matches `TS_INTERVAL_S` in Script 3)
