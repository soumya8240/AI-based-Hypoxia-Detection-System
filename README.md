# 🩺 IoT-Enabled Edge AI Hypoxia Detection and Alert System using Raspberry Pi

![Project Status](https://img.shields.io/badge/status-prototype-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-green)
![Microcontroller](https://img.shields.io/badge/microcontroller-Arduino%20UNO-teal)
![AI Model](https://img.shields.io/badge/model-Hybrid%201D--CNN%20%2B%20LSTM-orange)
![Edge AI](https://img.shields.io/badge/edge%20AI-TensorFlow%20Lite-red)
![IoT](https://img.shields.io/badge/cloud-ThingSpeak-lightgrey)
![Use](https://img.shields.io/badge/use-academic%20prototype-purple)

---

## 📌 Overview

This project presents a **low-cost, IoT-enabled Edge AI hypoxia detection and alert system** using a dual-processor architecture:

- **Arduino UNO** for sensor acquisition and local actuation  
- **Raspberry Pi 5** for edge AI inference, signal filtering, data logging, and cloud communication  

The system continuously monitors key physiological signals:

- 🫁 **SpO₂** — blood oxygen saturation  
- ❤️ **Heart Rate**  
- 🌡️ **Body Temperature**  
- 📈 **Perfusion Index (PI)**  

A **Hybrid 1D-CNN + LSTM** neural network model runs on the Raspberry Pi 5 using **TensorFlow Lite**. The model analyzes 30-second sliding windows of physiological data and classifies hypoxia risk into:

- 🟢 **Normal**
- 🟡 **Warning**
- 🔴 **Critical**

The system also applies **4-channel Kalman filtering** to suppress sensor noise and motion artifacts before inference. Local alerts are delivered through LEDs and a buzzer, while vitals and prediction outputs are uploaded to a **ThingSpeak 6-field cloud dashboard**.

> ⚠️ **Medical Disclaimer:**  
> This system is an academic proof-of-concept prototype. It is **not a certified medical device** and must not be used for clinical diagnosis, treatment, or medical decision-making without proper validation, safety testing, and regulatory approval.

---

## 🎯 Project Aim

The aim of this project is to build an affordable, real-time, edge-deployable hypoxia risk monitoring system that can provide early warning before a critical desaturation event.

The system is designed for potential future use in:

- 👶 Neonatal ICU monitoring
- 🏥 Rural and low-resource healthcare centers
- 🏠 Home care for chronic respiratory patients
- 👵 Elderly care and assisted living
- 🎓 Biomedical IoT and Edge AI education

---

## 🚀 Key Contributions

- 🏥 Uses **BIDMC PPG and Respiration Dataset** from PhysioNet containing 53 clinical ICU recordings
- 🤖 Implements a **Hybrid 1D-CNN + LSTM** model for temporal physiological pattern recognition
- ⏱️ Uses **30-second sliding windows** with 50% overlap
- 🔮 Applies **look-ahead temporal labelling** to identify pre-hypoxic warning patterns 2–5 minutes before critical events
- 📊 Uses 4-feature physiological fusion: **SpO₂ + Heart Rate + Temperature + Perfusion Index**
- 🧹 Applies **4-channel scalar Kalman filtering** for noise and motion-artifact suppression
- 🔌 Uses a robust **Arduino UNO + Raspberry Pi 5** dual-processor architecture
- 🔁 Uses USB serial communication at **115200 baud**
- ☁️ Uploads 6 fields to **ThingSpeak** with retry logic
- ⚡ Supports TensorFlow Lite deployment using built-in operations only
- 🧪 Includes simulator support for testing without physical hardware

---

## 🏗️ System Architecture

The overall system follows this pipeline:

```text
Sensors
   ↓
Arduino UNO
Sensor Acquisition + Initial Filtering + Actuation
   ↓
USB Serial Communication
   ↓
Raspberry Pi 5
Kalman Filtering + Sliding Window + AI Inference
   ↓
Alert System + CSV Logging + ThingSpeak Dashboard
```

### Architecture Summary

| Layer | Function |
|---|---|
| Sensor Layer | Acquires SpO₂, heart rate, temperature, and PPG-derived PI |
| Arduino Layer | Reads sensors, smooths signals, streams data, controls LEDs/buzzer |
| Raspberry Pi Layer | Applies Kalman filtering, runs AI model, logs data, updates cloud |
| AI Layer | Hybrid 1D-CNN + LSTM model classifies risk |
| Alert Layer | Green, yellow, red LEDs and buzzer provide local feedback |
| Cloud Layer | ThingSpeak dashboard displays vitals and risk status |

---

## 🔧 Hardware Components

| # | Component | Qty | Approx. Cost (INR) | Purpose |
|---:|---|---:|---:|---|
| 1 | Raspberry Pi 5, 4 GB / 8 GB | 1 | ₹5,500–₹7,500 | Edge AI compute |
| 2 | Arduino UNO | 1 | ₹500–₹800 | Sensor acquisition and actuation |
| 3 | MAX30102 Pulse Oximeter | 1 | ₹150–₹300 | SpO₂, heart rate, PI |
| 4 | DS18B20 Temperature Sensor | 1 | ₹80–₹150 | Body-surface temperature |
| 5 | Active Buzzer, 5 V | 1 | ₹20–₹40 | Audible alert |
| 6 | LEDs: Green, Yellow, Red | 3 | ₹10 | Visual risk indicators |
| 7 | 220 Ω Resistors | 3 | ₹5 | LED current limiting |
| 8 | 4.7 kΩ Resistor | 1 | ₹5 | DS18B20 pull-up |
| 9 | USB-A to USB-B Cable | 1 | ₹100 | Arduino–Raspberry Pi link |
| 10 | Breadboard + Jumper Wires | 1 set | ₹100 | Prototyping |
| 11 | MicroSD Card, 128 GB | 1 | ₹600 | Raspberry Pi OS |
| 12 | USB-C Power Supply, 5 V / 5 A | 1 | ₹500 | Raspberry Pi 5 power |

💰 **Estimated total cost:** ₹7,700–₹10,100

---

## 💻 Software Stack

| Layer | Technology |
|---|---|
| Raspberry Pi Programming | Python 3.11+ |
| Arduino Programming | C++ / Arduino IDE |
| AI Training | TensorFlow / Keras 2.x |
| Edge Inference | TensorFlow Lite Runtime |
| Signal Processing | SciPy, NumPy, Butterworth filters |
| Noise Suppression | Custom scalar Kalman filter |
| Serial Communication | pyserial, USB CDC/ACM, 115200 baud |
| Cloud Platform | ThingSpeak REST API |
| Data Logging | CSV: raw and Kalman-filtered vitals |
| Dataset Source | BIDMC PPG and Respiration Dataset, PhysioNet |
| Dataset Processing | wfdb, pandas, scipy |
| Arduino Libraries | SparkFun MAX3010x, OneWire, DallasTemperature |

---

## 🔌 Arduino UNO Pin Mapping

| Function | Arduino Pin | Mode | Notes |
|---|---|---|---|
| MAX30102 SDA | A4 | I²C Data | On-board 4.7 kΩ pull-ups |
| MAX30102 SCL | A5 | I²C Clock | On-board 4.7 kΩ pull-ups |
| DS18B20 Data | D4 | 1-Wire Data | 4.7 kΩ pull-up to 5 V |
| Green LED | D5 | Digital Output | 220 Ω resistor to GND |
| Yellow LED | D6 | Digital Output / PWM | 220 Ω resistor, PWM pulsing |
| Red LED | D7 | Digital Output | 220 Ω resistor to GND |
| Active Buzzer | D8 | Digital Output | Direct connection |

---

## 🔁 USB Serial Configuration

| Parameter | Value |
|---|---|
| Cable Type | USB-A from Raspberry Pi to USB-B from Arduino |
| Baud Rate | 115200 |
| Device Path, Genuine Arduino | `/dev/ttyACM0` |
| Device Path, CH340 Clone | `/dev/ttyUSB0` |
| Sensor Data Format | `"SpO2,HR,Temp,PI\n"` |
| Data Frequency | 1 Hz |
| Command Format | Single byte: `N`, `W`, or `C` |

### Serial Commands

| Command | Meaning | Arduino Action |
|---|---|---|
| `N` | Normal | Green LED ON |
| `W` | Warning | Yellow LED PWM pulsing |
| `C` | Critical | Red LED ON + buzzer ON |

---

## 🧠 AI Model

The updated system uses a **Hybrid 1D-CNN + LSTM** architecture designed to capture both:

- CNN-based local/morphological temporal patterns
- LSTM-based sequential physiological trends

### Input

Each input sample is a 30-second window:

```text
Shape: (30, 4)

Features:
[SpO₂, Heart Rate, Temperature, Perfusion Index]
```

### Output Classes

```text
0 → Normal
1 → Warning
2 → Critical
```

### Model Architecture

| Layer | Type | Output Shape | Details |
|---|---|---|---|
| Input | Input | `(30, 4)` | 30-second window, 4 features |
| Conv1 | Conv1D | `(30, 64)` | 64 filters, kernel=5, causal, ReLU |
| BN1 | BatchNorm | `(30, 64)` | Batch normalization |
| Pool1 | MaxPool1D | `(15, 64)` | Pool size = 2 |
| Conv2 | Conv1D | `(15, 128)` | 128 filters, kernel=3, causal, ReLU |
| BN2 | BatchNorm | `(15, 128)` | Batch normalization |
| Pool2 | MaxPool1D | `(7, 128)` | Pool size = 2 |
| Conv3 | Conv1D | `(7, 64)` | 64 filters, kernel=3, causal, ReLU |
| BN3 | BatchNorm | `(7, 64)` | Batch normalization |
| LSTM1 | LSTM | `(7, 64)` | 64 units, return sequences, unroll=True |
| Dropout1 | Dropout | `(7, 64)` | Rate = 0.1 |
| LSTM2 | LSTM | `(32,)` | 32 units, unroll=True |
| FC1 | Dense | `(64,)` | ReLU |
| Dropout2 | Dropout | `(64,)` | Rate = 0.35 |
| FC2 | Dense | `(32,)` | ReLU |
| Dropout3 | Dropout | `(32,)` | Rate = 0.2 |
| Output | Dense | `(3,)` | Softmax |

### TensorFlow Lite Compatibility

The LSTM layers use:

```python
unroll=True
```

and avoid:

```python
recurrent_dropout
```

This prevents dynamic TensorList conversion errors and allows the model to convert to TensorFlow Lite using **BUILTINS-only operations**, avoiding the need for a Flex delegate on Raspberry Pi.

---

## 📊 Dataset

The system is trained using the **BIDMC PPG and Respiration Dataset** from PhysioNet.

### Dataset Details

- 53 clinical ICU recordings
- 125 Hz photoplethysmogram waveform
- 1 Hz numeric channels:
  - SpO₂
  - Heart Rate

### Derived Features

#### Perfusion Index

Perfusion Index is calculated from the raw PPG signal:

```text
PI = (ACpeak-to-peak / DCmean) × 100%
```

where:

- `AC` is the bandpass-filtered pulsatile cardiac component, 0.5–4 Hz
- `DC` is the lowpass-filtered baseline component, < 0.08 Hz

#### Temperature

Because BIDMC does not include body temperature, temperature is physiologically simulated using:

- Peripheral vasoconstriction model during hypoxia
- Approximate 0.05°C drop per 1% SpO₂ below 95%
- Autoregressive drift
- DS18B20-like Gaussian noise, σ = 0.08°C

---

## 🏷️ Risk Labelling Strategy

The system uses a 3-pass look-ahead temporal labelling method.

### Pass 1: Critical Detection

Mark time steps as **Critical** if:

```text
SpO₂ < 90%
OR HR < 50 bpm
OR HR > 120 bpm
OR PI < 0.2%
```

### Pass 2: Warning Look-Back

For every Critical event at time `t`, mark:

```text
[t - 300 seconds, t - 120 seconds]
```

as **Warning**, representing a 2–5 minute pre-hypoxic window.

### Pass 3: Physiological Confirmation

Retain Warning labels only when physiological warning signs are present, such as:

```text
SpO₂ = 90–95%
borderline HR
borderline PI
```

---

## 🩸 Clinical Thresholds for Labelling

| Risk Class | SpO₂ (%) | Heart Rate (bpm) | PI (%) | Temperature (°C) |
|---|---|---|---|---|
| 🟢 Normal | ≥ 95 | 60–100 | ≥ 0.5 | 36.1–37.5 |
| 🟡 Warning | 90–94 | 55–110 borderline | 0.2–0.49 | 35.5–38.5 |
| 🔴 Critical | < 90 | < 50 or > 120 | < 0.2 | < 35.5 or > 38.5 |

---

## 📐 Feature Normalization

Features are clipped and normalized to `[0, 1]`.

| Feature | Min | Max |
|---|---:|---:|
| SpO₂ (%) | 70.0 | 100.0 |
| Heart Rate (bpm) | 20.0 | 200.0 |
| Temperature (°C) | 35.5 | 39.5 |
| Perfusion Index (%) | 0.0 | 25.0 |

Normalization formula:

```text
x_norm = (x - x_min) / (x_max - x_min)
```

---

## 🧹 Kalman Filtering

Four independent scalar Kalman filters are applied on the Raspberry Pi, one for each physiological channel:

- SpO₂
- Heart Rate
- Temperature
- Perfusion Index

### Kalman Noise Parameters

| Channel | Q, Process Noise | R, Measurement Noise | Rationale |
|---|---:|---:|---|
| SpO₂ | 0.05 | 0.5 | Slowly varying; suppress motion spikes |
| Heart Rate | 0.5 | 2.0 | Moderate variability; noisy raw signal |
| Temperature | 0.005 | 0.01 | Very stable; high sensor accuracy |
| Perfusion Index | 0.02 | 0.1 | Highly susceptible to motion |

---

## 🧪 Data Augmentation

To address class imbalance, augmentation is applied to minority-class training windows.

### Techniques Used

1. **Gaussian noise injection**  
   Additive noise with σ = 0.01

2. **Amplitude jitter**  
   ±5% per-feature multiplicative scaling

3. **Time warp**  
   Random time-stretching/compression from 90–110% using interpolation

🎯 Target: each minority class reaches at least 40% of the majority class count.

---

## 📦 TensorFlow Lite Edge Deployment

The TensorFlow Lite model is exported using a cascading fallback strategy:

1. **Full-integer Int8 quantization**  
   Smallest and fastest option

2. **Dynamic-range quantization**  
   Int8 weights with float activations

3. **Float32 model**  
   Safest fallback, built-in operations only

On Raspberry Pi 5, inference is performed using:

```python
tflite_runtime
```

### Edge Inference Loop

```text
Parse serial data
   ↓
Apply Kalman filter
   ↓
Update 30-second sliding window
   ↓
Normalize features
   ↓
Run TensorFlow Lite inference
   ↓
Send command to Arduino
   ↓
Log CSV data
   ↓
Upload to ThingSpeak
```

---

## ☁️ ThingSpeak Dashboard

The Raspberry Pi uploads 6 fields to ThingSpeak every 15 seconds.

### Uploaded Fields

| Field | Data |
|---|---|
| Field 1 | SpO₂ |
| Field 2 | Heart Rate |
| Field 3 | Temperature |
| Field 4 | Perfusion Index |
| Field 5 | Risk Level |
| Field 6 | Confidence Percentage |

The dashboard displays:

- Real-time numeric values
- Gauges
- Line charts
- Risk-level indicators

ThingSpeak upload includes **exponential backoff retry logic** with up to 3 attempts.

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Training Accuracy | > 98% |
| Inference Time on Raspberry Pi 5 | < 10 ms per prediction |
| Model File Size | Varies by quantization strategy |
| RAM Usage During Inference | ~2–4 MB |
| Quantization Strategy | Int8 → Dynamic → Float32 |
| TensorFlow Lite Ops | BUILTINS only, no Flex delegate |

---

## ⚡ System Performance

| Parameter | Value |
|---|---|
| Sensor Sampling Rate | 1 Hz, Arduino to Raspberry Pi |
| Kalman Filter Latency | < 1 ms per sample per channel |
| Sliding Window Size | 30 samples, 30 seconds |
| End-to-End Inference Time | < 10 ms per window |
| Cloud Update Rate | Every 15 seconds |
| Alert Response Time | < 100 ms |
| USB Serial Reliability | Auto-reconnect on disconnection |
| ThingSpeak Retry | Exponential backoff, up to 3 attempts |

---

## 🧪 Simulation Environment

A dedicated Python-based simulator is included:

```text
simulator/simulate_vitals.py
```

It generates realistic vital-sign streams with gradual physiological transitions between:

- Normal
- Warning
- Critical

Supported scenarios include:

- Random
- Critical
- Declining

The simulator can feed data directly into the main pipeline for full software testing without physical hardware.

---

## 🔬 LTspice Circuit Simulation

The alert unit was simulated in LTspice before hardware implementation.

### Buzzer Circuit

- Uses a **2N2222 NPN transistor** as a low-side switch
- Base driven through a 1 kΩ resistor
- Buzzer switches ON/OFF based on pulse input

### LED Circuit

- Green LED: Normal
- Yellow LED: Warning
- Red LED: Critical
- Each LED uses a 220 Ω current-limiting resistor

The simulations verified that the Arduino/Raspberry Pi control signals can safely switch local alert indicators.

---

## 🧰 Hardware Prototype

The prototype was assembled on a breadboard using:

- Arduino UNO
- Raspberry Pi 5
- MAX30102 sensor
- DS18B20 temperature sensor
- Active buzzer
- Tri-color LEDs

Hardware checks included:

```bash
ls /dev/ttyACM0
```

The MAX30102 was verified at I²C address:

```text
0x57
```

The DS18B20 was confirmed on the 1-Wire bus.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/iot-edge-ai-hypoxia-detection.git
cd iot-edge-ai-hypoxia-detection
```

### 2. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
numpy
pandas
scipy
wfdb
tensorflow
tflite-runtime
pyserial
requests
python-dotenv
```

### 4. Configure ThingSpeak

Create a `.env` file:

```bash
THINGSPEAK_API_KEY=your_thingspeak_write_api_key
THINGSPEAK_CHANNEL_ID=your_channel_id
```

### 5. Upload Arduino Code

Use Arduino IDE to upload the Arduino sensor-acquisition and alert-control code.

Required Arduino libraries:

```text
SparkFun MAX3010x
OneWire
DallasTemperature
```

---

## ▶️ Running the System

### Run the Main Raspberry Pi Pipeline

```bash
python main.py
```

### Run the Vital Signs Simulator

```bash
python simulator/simulate_vitals.py
```

### Check Arduino Serial Port

```bash
ls /dev/ttyACM0
ls /dev/ttyUSB0
```

### Example Serial Data Format

```text
98.2,82,36.7,1.24
```

### Example Risk Commands Sent to Arduino

```text
N
W
C
```

---

## 📁 Suggested Repository Structure

```text
iot-edge-ai-hypoxia-detection/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── arduino/
│   └── hypoxia_sensor_alert_node.ino
│
├── src/
│   ├── main.py
│   ├── serial_reader.py
│   ├── kalman_filter.py
│   ├── sliding_window.py
│   ├── inference.py
│   ├── thingspeak_upload.py
│   └── logger.py
│
├── model/
│   ├── hypoxia_cnn_lstm_int8.tflite
│   ├── hypoxia_cnn_lstm_dynamic.tflite
│   └── hypoxia_cnn_lstm_float32.tflite
│
├── training/
│   ├── preprocess_bidmc.py
│   ├── extract_perfusion_index.py
│   ├── create_labels.py
│   ├── train_cnn_lstm.py
│   └── export_tflite.py
│
├── simulator/
│   └── simulate_vitals.py
│
├── hardware/
│   ├── wiring_diagram.png
│   ├── buzzer_ltspice.asc
│   └── led_ltspice.asc
│
├── data/
│   └── sample_logs.csv
│
└── tests/
    └── test_pipeline.py
```

---

## 🧯 Challenges Faced

The following challenges were encountered and addressed:

1. **Raspberry Pi OS flashing issue**  
   Resolved by reverting to a previous version of Raspberry Pi Imager.

2. **PC interfacing issue**  
   Switching from Windows to Linux provided more stable Raspberry Pi communication.

3. **Sensor garbage values**  
   Resolved using moving-average filtering, Kalman filtering, and 5–10 second sensor warm-up.

4. **ThingSpeak upload failure**  
   Fixed by correcting HTTP POST headers and adding exponential backoff retry logic.

5. **Dataset transition**  
   Migration from synthetic data to BIDMC clinical data required development of a complete signal-processing pipeline.

6. **TFLite LSTM conversion issue**  
   Resolved by using `unroll=True`, removing `recurrent_dropout`, and avoiding dynamic TensorList operations.

7. **TFLite runtime conflicts on Raspberry Pi**  
   Addressed through a quantization fallback strategy.

8. **USB serial auto-detection**  
   Implemented support for both `/dev/ttyACM0` and `/dev/ttyUSB0`.

9. **I²C pin detection error**  
   Resolved by verifying wiring and using Arduino’s built-in I²C scanner.

---

## ✅ Conclusions

This project demonstrates that an affordable **Edge AI-based hypoxia monitoring and early-warning system** can be developed using open-source tools and low-cost hardware.

Key achievements include:

- Training on real-world BIDMC PhysioNet clinical ICU recordings
- Hybrid 1D-CNN + LSTM temporal modelling
- 30-second sliding-window inference
- 2–5 minute pre-hypoxic warning strategy
- 4-feature physiological fusion
- Kalman-filter-based noise suppression
- Dual-processor Arduino + Raspberry Pi deployment
- ThingSpeak-based remote dashboard
- Local LED and buzzer alerting

The system is not clinical-grade yet, but it provides a strong foundation for future clinical validation and wearable health-monitoring development.

---

## ⚠️ Limitations

- Temperature is simulated for the BIDMC dataset because BIDMC does not include body temperature.
- MAX30102 die temperature is not equivalent to core body temperature.
- Single-sensor design limits broader clinical assessment.
- ThingSpeak free tier limits upload frequency to once every 15 seconds.
- The model has not yet been validated in a prospective clinical trial.

---

## 🔮 Future Scope

- 🏥 Validate the model using prospective patient data with IRB-approved clinical studies
- 📚 Integrate larger datasets such as MIMIC-IV
- ⚡ Replace HTTP REST with MQTT for lower-latency cloud communication
- 🫁 Add respiration-rate estimation from PPG waveform analysis
- 📱 Add SMS, email, or mobile notifications for caregivers
- 🧠 Explore EEG or EOG-based assistive technology extensions
- 🔋 Build a compact battery-powered wearable version using ESP32 or Pi Zero 2W
- 🔐 Implement federated learning for privacy-preserving multi-device model updates
- ⏱️ Further improve early trend detection with temporal deep learning

---

## 👨‍💻 Authors

- **Soumya Ray**
- **Urnisa Rakshit**

---

## 📚 References

1. S. Joo et al., “A Patient Management System Using an Edge Computing-Based IoT Pulse Oximeter,” IEEE Access, 2024.
2. Z. Ashfaq et al., “Embedded AI-Based Digi-Healthcare,” Frontiers in Public Health, 2022.
3. PhysioNet, “BIDMC PPG and Respiration Dataset.”
4. PhysioNet, “MIMIC-IV Clinical Database.”
5. World Health Organization, *Pulse Oximetry Training Manual*, Geneva, Switzerland, 2011.
6. H. R. Graham et al., “Reducing global inequities in medical oxygen access,” *The Lancet Global Health*, 2025.
7. British Thoracic Society, “BTS Guideline for Oxygen Use in Adults in Healthcare and Emergency Settings,” BMJ Respiratory, 2017.
8. TensorFlow, “TensorFlow Lite.”
9. ThingSpeak, “ThingSpeak IoT Platform.”
10. Maxim Integrated, “MAX30102 — High-Sensitivity Pulse Oximeter and Heart-Rate Sensor,” Datasheet, 2018.
11. R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” *Journal of Basic Engineering*, 1960.

---

## ⭐ Support

If this project is useful, consider giving the repository a star ⭐.

---
