# Techies
# 🛡️ IoT & ML-based Worker Safety Monitoring System

![GitHub Stars](https://img.shields.io/github/stars/your-username/your-repo-name?style=social)
![Hackathon Winner](https://img.shields.io/badge/Hackathon-Winner-green)
![License](https://img.shields.io/github/license/your-username/your-repo-name)

A real-time safety and environmental monitoring system designed to **detect harmful asbestos particles**, track **vital signs of factory workers**, and ensure **occupational safety** in high-risk industrial zones like **sugar mills**. Powered by **IoT**, **ML**, and **Blynk/Firebase Dashboard Integration**, this project won accolades at HackOnHills 6.0.

---

## 🚀 Overview

**Problem:** Sugar mill workers are exposed to high levels of asbestos and fluctuating environmental conditions, which can lead to serious health hazards including mesothelioma.

**Solution:** An integrated IoT and ML system that:
- Monitors **asbestos** levels (via MQ sensors)
- Tracks **heart rate**, **body temperature**, and **environmental conditions**
- Sends **real-time alerts** via a **dashboard (Blynk/Firebase + Node.js)**
- Uses **machine learning** to analyze risk levels

---

## 🏆 Highlights

- 🧠 **Asbestos Detection using ML**
- 💡 **Real-time alerting system (LED + buzzer)**
- 📊 **Interactive dashboard for remote monitoring**
- 📶 **Wireless data transmission using Wi-Fi/Bluetooth**
- 🔧 **Custom-built wearable device for workers**
- 🌐 **Web-based monitoring using Node.js & Firebase**

---

## 📸 Demo & Screenshots

| Hardware Setup | Dashboard View |
|----------------|----------------|
| ![hardware](assets/hardware.jpg) | ![dashboard](assets/dashboard.jpg) |

*Replace with your real image links or paths.*

---

## 🔍 Tech Stack

| Tech | Role |
|------|------|
| **Arduino / Vega Aries V3** | Microcontroller-based data collection |
| **MQ135, DHT22, HW827** | Sensor suite (Asbestos, Temp, Heart rate) |
| **Python** | ML model for anomaly detection |
| **Node.js + Firebase** | Real-time web dashboard |
| **Blynk** *(Optional)* | IoT remote monitoring mobile app |
| **HTML/CSS/JS** | Frontend dashboard UI |

---

## 💻 How It Works

1. **Sensor Kit** collects real-time data (air quality, vitals, temp).
2. **Microcontroller** processes and sends data wirelessly.
3. **ML Algorithm** detects harmful patterns or levels.
4. **Buzzer + LED** alerts workers on-site.
5. **Dashboard** displays live readings for supervisors remotely.

---

## 🧠 Machine Learning

- **Model**: Trained using collected asbestos data using SVM/Random Forest.
- **Function**: Classifies readings into **Safe**, **Moderate**, and **Critical** zones.
- **Dataset**: Self-generated via calibrated sensor readings.

---

## 📦 Setup Instructions

### 🧰 Hardware Required
- Vega Aries V3 Board or Arduino
- MQ135 + MQ5 Gas Sensors
- DHT22 (Temperature + Humidity)
- HW-827 (Heart Rate)
- Piezo Buzzer + LED
- Breadboard & jumper wires

### 💾 Software Setup

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

