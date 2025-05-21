# Hand Gesture Recognition with CNN – Interactive Real-Time Detection App

**Class: MIV1 S2 2024/2025 – Dr. Sebai**

## 📌 Project Overview

This project explores the design and deployment of a real-time hand gesture recognition system using **Convolutional Neural Networks (CNNs)**. The final application allows users to **assign custom actions** to recognized hand gestures, enabling an interactive control system through camera-captured movements.

The core objective is to transform hand movements—especially those of the **right hand**—into actionable commands for interactive environments, opening the door to creative use cases such as game control, UI interaction, or general system commands.

## 🎯 Objectives

* **Build a robust CNN model** capable of recognizing distinct right-hand movements from live camera input.
* **Develop an interactive application** that connects gesture detection to customizable commands.
* **Ensure real-time performance** through optimized model design and system integration.
* **Encourage flexibility** in gesture-action mapping, so users can tailor their experience.

## 🧠 Project Flow

This project was developed in a small team (binôme/trinôme) as part of the MIV1 curriculum, under the guidance of **Dr. Sebai**. Throughout the process, we experimented with different architectures and data-handling strategies, ultimately choosing a CNN-based approach inspired by recent research ([referenced paper](https://arxiv.org/pdf/2309.11610)).

### Key Steps:

1. **Research & Exploration**

   * Started from an academic paper on gesture detection.
   * Extended our study to include related models, data preprocessing techniques, and architecture optimization.

2. **Model Development**

   * Built and trained a CNN model using gesture datasets.
   * Focused on right-hand gesture classification for real-time responsiveness.

3. **System Integration**

   * Developed a lightweight **Flask app** enhanced with **WebSocket** support for low-latency communication.
   * Integrated the model into the app, enabling live gesture prediction via webcam.

4. **Interactive Mapping**

   * Implemented an interface allowing gestures to be assigned freely to specific actions.
   * Created a scalable system that can be extended to control various applications or devices.

## 🛠️ Tech Stack

* **Python** / **Flask**
* **OpenCV** – Video capture and frame processing
* **TensorFlow / Keras** – CNN model implementation
* **Socket.IO** – Real-time bidirectional communication
* **JavaScript / HTML / CSS** – Frontend interactivity

## 🧪 How It Works

* User starts the web app (Flask + WebSocket).
* Camera captures hand movement in real-time.
* The model processes video frames and classifies gestures.
* The app interprets gestures as actions and provides immediate feedback or performs mapped tasks.
