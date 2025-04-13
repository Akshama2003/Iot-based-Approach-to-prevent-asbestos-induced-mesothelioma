# Techies
# 🛡️ IoT & ML-based Worker Safety Monitoring System

![GitHub Stars](https://img.shields.io/github/stars/your-username/your-repo-name?style=social)
![License](https://img.shields.io/github/license/your-username/your-repo-name)
![WhatsApp Image 2025-04-13 at 12 49 00_4c74732e](https://github.com/user-attachments/assets/1ecbb706-ccc0-411e-a6ad-db2b2f0bc302)
![WhatsApp Image 2025-04-13 at 12 49 00_bfe5bb03](https://github.com/user-attachments/assets/c58f64a3-368d-438b-b7a1-358d5fbf2524)
![https://drive.google.com/file/d/1cLq2hza6J_DlHjaHBmOfTW81Ugexq88C/view?usp=drivesdk ]


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


![Screenshot (42)](https://github.com/user-attachments/assets/f8a05dce-bd1d-4d96-b472-82a14dfe19b9)
l image links or paths.*![Screenshot (43)](https://github.com/user-attachments/assets/cd6f8263-debf-4762-b4b2-da50afd91e36)

![Screenshot (44)](https://github.com/user-attachments/assets/233b0a5c-9f23-41db-9001-9784083f3aa0)
![Screenshot (45)](https://github.com/user-attachments/assets/3774c76b-cd80-4baf-8de4-551195e69efe)
![Screenshot (46)](https://github.com/user-attachments/assets/000ca244-c47f-49fd-8558-9902ee690a2a)

---![Screenshot (41)](https://github.com/user-attachments/assets/8aa6e2bd-7ee2-451b-8b12-b3758c9c96be)

![Screenshot (47)](https://github.com/user-attachments/assets/0ce7b4a9-8ca5-4efe-9b13-1d7709d5e583)

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

