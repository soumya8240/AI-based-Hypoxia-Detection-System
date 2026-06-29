# 🩺 IoT-Enabled Edge AI Hypoxia Detection and Alert System using Raspberry Pi

![Project Status](https://img.shields.io/badge/status-prototype-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![AI](https://img.shields.io/badge/AI-TensorFlow%20Lite-orange)
![IoT](https://img.shields.io/badge/IoT-ThingSpeak-lightgrey)
![License](https://img.shields.io/badge/license-Academic%20Use-purple)

---

## 📌 Overview

This project presents a **low-cost, IoT-enabled, edge AI-based hypoxia detection and alert system** designed for real-time physiological monitoring.

The system continuously measures key vital signs, including:

- 🫁 Blood oxygen saturation (**SpO₂**)
- ❤️ Heart rate
- 🌡️ Body temperature

Using an on-device machine learning model deployed on a **Raspberry Pi 5**, the system classifies hypoxia risk into three levels:

- 🟢 **Normal**
- 🟡 **Warning**
- 🔴 **Critical**

The system integrates affordable biomedical sensors, local alert mechanisms, cloud-based monitoring, and a real-time dashboard. It is intended as an educational and proof-of-concept prototype for affordable health monitoring in resource-limited environments such as rural healthcare centers, home-care settings, elderly care, and neonatal intensive care monitoring.

> ⚠️ **Important Disclaimer:**  
> This project is a prototype and is **not intended for clinical diagnosis or medical decision-making** without further validation, calibration, regulatory approval, and clinical testing.

---

## 🚀 Key Features

- 🩺 Real-time monitoring of SpO₂, heart rate, and body temperature
- 🤖 Edge AI-based hypoxia risk prediction using TensorFlow Lite
- 🧠 Lightweight machine learning model deployable on Raspberry Pi
- 🟢🟡🔴 Three-class risk classification: Normal, Warning, and Critical
- 🔊 Local alert system using buzzer and tri-color LED indicators
- ☁️ Cloud data upload to ThingSpeak for remote monitoring
- 📊 Local web dashboard for real-time vitals visualization
- 💰 Low-cost hardware design using Raspberry Pi 5 and affordable sensors
- 🧪 Simulation environment for testing without physical hardware
- ⚡ Fast inference time of less than 5 ms on Raspberry Pi 5

---

## 🎯 Project Motivation

Hypoxia occurs when body tissues do not receive sufficient oxygen. It can lead to serious clinical deterioration, organ damage, and death if not detected early.

Conventional monitoring systems are often expensive, cloud-dependent, and may introduce latency. These limitations make them less suitable for low-resource or remote healthcare settings.

This project aims to demonstrate that an affordable **edge AI system** can continuously monitor physiological signals and provide early warnings for hypoxia-related risk without depending entirely on cloud processing.

---

## 🏗️ System Architecture

The system follows a modular pipeline:

```text
Sensors → Data Acquisition → Preprocessing → Edge AI Inference → Alert System → Cloud Dashboard
```

### 🔧 Main Components

1. **Sensing Layer**
   - MAX30102 pulse oximeter sensor for SpO₂ and heart rate
   - DS18B20 digital temperature sensor for body temperature

2. **Edge Computing Layer**
   - Raspberry Pi 5 processes sensor data
   - TensorFlow Lite model performs real-time hypoxia risk prediction

3. **Alert Layer**
   - Green LED: Normal condition
   - Yellow LED: Warning condition
   - Red LED + buzzer: Critical condition

4. **Cloud and Dashboard Layer**
   - ThingSpeak dashboard for remote monitoring
   - Flask-based local dashboard for real-time vitals visualization

---

## 🧰 Hardware Components

| Component | Quantity | Purpose |
|---|---:|---|
| Raspberry Pi 5 | 1 | Edge computing and AI inference |
| MAX30102 Pulse Oximeter Module | 1 | SpO₂ and heart-rate measurement |
| DS18B20 Temperature Sensor | 1 | Body temperature measurement |
| Active Buzzer | 1 | Audible alert |
| Green, Yellow, Red LEDs | 3 | Visual risk indicators |
| 220Ω Resistors | 3 | LED current limiting |
| BC547B NPN Transistor | 1 | Buzzer driver circuit |
| 4.7kΩ Resistor | 1 | DS18B20 pull-up resistor |
| Breadboard and Jumper Wires | 1 set | Prototyping |
| MicroSD Card | 1 | Raspberry Pi OS storage |
| USB-C Power Supply | 1 | Raspberry Pi power supply |

💸 **Approximate total cost:** ₹7,100 – ₹9,200

---

## 💻 Software Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.11+ |
| AI Framework | TensorFlow / Keras |
| Edge Inference | TensorFlow Lite Runtime |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Chart.js |
| Cloud Platform | ThingSpeak REST API |
| Hardware Interface | smbus2, RPi.GPIO |
| Testing | Vital signs simulator, pytest |
| Configuration | python-dotenv |

---

## 🔌 GPIO Pin Mapping

| Function | GPIO Pin | Physical Pin | Mode | Notes |
|---|---:|---:|---|---|
| MAX30102 SDA | GPIO 2 | Pin 3 | I²C Data | I²C Bus 1 |
| MAX30102 SCL | GPIO 3 | Pin 5 | I²C Clock | I²C Bus 1 |
| DS18B20 Data | GPIO 4 | Pin 7 | 1-Wire Data | 4.7kΩ pull-up to 3.3V |
| Buzzer | GPIO 18 | Pin 12 | PWM Output | Driven via NPN transistor |
| Green LED | GPIO 23 | Pin 16 | Digital Output | Normal state |
| Yellow LED | GPIO 24 | Pin 18 | Digital Output | Warning state |
| Red LED | GPIO 25 | Pin 22 | Digital Output | Critical state |

---

## 🩸 Risk Classification Logic

The system classifies physiological status into three risk categories.

| Risk Class | SpO₂ Range | Heart Rate | Temperature |
|---|---|---|---|
| 🟢 Normal | ≥ 95% | 60–100 bpm | 36.1–37.2°C |
| 🟡 Warning | 90–94% | Mild deviation | Mild deviation |
| 🔴 Critical | < 90% | Severe deviation | Severe deviation |

---

## 🤖 AI Model

A lightweight neural network model is trained to classify patient risk status based on physiological input features.

### 📥 Input Features

```text
[SpO₂, Heart Rate, Body Temperature]
```

### 🧠 Model Architecture

```text
Dense(32, ReLU)
Dense(16, ReLU)
Dense(3, Softmax)
```

### 📤 Output Classes

The model produces a three-class probability distribution:

```text
0 → Normal
1 → Warning
2 → Critical
```

The class with the highest probability is selected as the final risk status, and the confidence score is displayed on the dashboard.

### 📦 Edge Deployment

The trained model is converted into TensorFlow Lite format and deployed on Raspberry Pi 5 using `tflite-runtime`.

---

## 📈 Model and System Performance

| Metric | Value |
|---|---|
| Training Accuracy | > 98% |
| Inference Time on Raspberry Pi 5 | < 5 ms per prediction |
| Model File Size | ~10 KB |
| RAM Usage During Inference | ~2 MB |
| End-to-End Loop Time | ~2 seconds |
| Cloud Upload Rate | Every 15 seconds |
| Alert Response Time | < 100 ms |
| Stable Test Duration | 1 hour |

---

## 📊 Dashboard

The dashboard displays:

- 🫁 Current SpO₂ value
- ❤️ Heart rate
- 🌡️ Body temperature
- 🚦 Risk level
- 📈 Risk probability/confidence
- 📉 Live time-series plots
- 🟢🟡🔴 Color-coded alert banner

### Dashboard States

| State | Indicator | Meaning |
|---|---|---|
| 🟢 Normal | Green banner / LED | Safe physiological range |
| 🟡 Warning | Yellow banner / LED | Moderate hypoxia risk |
| 🔴 Critical | Red banner / LED + buzzer | High-risk condition requiring attention |

---

## ☁️ ThingSpeak Cloud Integration

The Raspberry Pi uploads real-time physiological data and model predictions to ThingSpeak using the REST API.

Typical fields include:

1. SpO₂
2. Heart Rate
3. Temperature
4. Risk Level
5. Risk Probability
6. Optional system/status field

ThingSpeak allows remote visualization of patient vitals and risk trends over time.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/iot-edge-ai-hypoxia-detection.git
cd iot-edge-ai-hypoxia-detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies may include:

```text
numpy
pandas
tensorflow
tflite-runtime
flask
requests
smbus2
RPi.GPIO
python-dotenv
pytest
```

### 4. Configure Environment Variables

Create a `.env` file:

```bash
THINGSPEAK_API_KEY=your_thingspeak_write_api_key
THINGSPEAK_CHANNEL_ID=your_channel_id
```

---

## ▶️ Running the Project

### Run with Hardware Sensors

```bash
python main.py
```

### Run in Simulation Mode

```bash
python simulator.py
```

### Start Local Flask Dashboard

```bash
python app.py
```

Then open the dashboard in your browser:

```text
http://<raspberry-pi-ip-address>:5000
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
├── hardware/
│   ├── gpio_pin_mapping.md
│   └── circuit_diagram.png
│
├── models/
│   ├── hypoxia_model.tflite
│   └── scaler.pkl
│
├── src/
│   ├── main.py
│   ├── sensor_reader.py
│   ├── inference.py
│   ├── alerts.py
│   ├── thingspeak_upload.py
│   └── config.py
│
├── dashboard/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── dashboard.js
│
├── simulation/
│   └── vital_signs_simulator.py
│
├── training/
│   ├── train_model.py
│   ├── generate_dataset.py
│   └── convert_to_tflite.py
│
└── tests/
    └── test_pipeline.py
```

---

## 🧪 Simulation Environment

A Python-based vital signs simulator is included to test the complete software pipeline without physical sensors.

The simulator can generate realistic physiological transitions between:

- Normal state
- Warning state
- Critical state

This allows validation of:

- AI inference
- Alert logic
- ThingSpeak upload
- Flask dashboard updates
- System response time

---

## 🧯 Challenges Faced

During development, the following challenges were encountered and addressed:

- Raspberry Pi OS flashing issues
- Sensor detection errors on the I²C bus
- Unstable sensor readings from MAX30102 and DS18B20
- ThingSpeak upload errors due to incorrect API request formatting
- TensorFlow Lite runtime installation conflicts on Raspberry Pi
- Need for synthetic data due to lack of a complete public dataset containing SpO₂, heart rate, and temperature together
- Hardware noise and warm-up instability during initial sensor readings

---

## ⚠️ Limitations

- The current model is trained on synthetic data and requires clinical validation.
- MAX30102 and DS18B20 readings are not equivalent to hospital-grade patient monitors.
- Body temperature from DS18B20 may not accurately reflect core body temperature.
- Single-device monitoring limits broader physiological assessment.
- ThingSpeak free tier limits upload frequency.
- The system is currently a prototype and should not be used for real medical diagnosis.

---

## 🔮 Future Scope

Planned improvements include:

- 🏥 Training and validating the model on real clinical datasets such as MIMIC-IV or other PhysioNet resources
- 🫁 Adding respiration-rate estimation from PPG waveform analysis
- ⚡ Replacing HTTP REST with MQTT for lower-latency cloud communication
- 📱 Adding SMS, email, or mobile app notifications for caregivers
- 🔋 Designing a compact wearable version with battery support
- 🧹 Improving motion-artifact removal using advanced filtering
- ⏱️ Extending the model to predict hypoxia several minutes before critical desaturation
- 🧬 Integrating additional sensors such as respiratory belt, ECG, or skin perfusion sensors

---

## 🌍 Applications

This project may be useful as a proof-of-concept for:

- 👶 Neonatal ICU monitoring
- 🏥 Rural healthcare centers
- 🏠 Home monitoring of respiratory patients
- 👵 Elderly care and assisted living
- 🎓 Low-cost biomedical IoT education
- 🤖 Edge AI-based healthcare prototyping

---

## ⚕️ Medical Disclaimer

This system is developed for **academic, educational, and proof-of-concept purposes only**.

It is **not a certified medical device** and must not be used for diagnosis, treatment, or clinical decision-making without proper medical validation, safety testing, and regulatory approval.

---

## 👨‍💻 Authors

- **Soumya Ray**
- **Urnisa Rakshit**

School of Artificial Intelligence  
Amrita Vishwa Vidyapeetham, Faridabad

---

## 🙏 Acknowledgement

We sincerely acknowledge the guidance and support of **Dr. Abhishek Kumar**, School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Faridabad, for his supervision, feedback, and encouragement throughout this project.

---

## 📚 References

1. Joo, S. et al. “A Patient Management System Using an Edge Computing-Based IoT Pulse Oximeter.” IEEE Access, 2024.
2. Ashfaq, Z. et al. “Embedded AI-Based Digi-Healthcare.” Frontiers in Public Health, 2022.
3. PhysioNet. MIMIC-IV Clinical Database.
4. World Health Organization. Pulse Oximetry Training Manual. Geneva: WHO, 2011.
5. TensorFlow Lite Documentation.
6. ThingSpeak IoT Platform Documentation.
7. Maxim Integrated. MAX30102 High-Sensitivity Pulse Oximeter and Heart-Rate Sensor Datasheet.

---

## 📜 License

This project is intended for academic and educational use.

You may add an appropriate license depending on your intended use, such as:

- MIT License
- Apache License 2.0
- GPL License

Example:

```text
MIT License
Copyright (c) 2026
```

---

## ⭐ Support

If you find this project useful, consider giving the repository a star ⭐.

---
