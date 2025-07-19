<h1 align="center">🖐️ Finger Counting with MediaPipe</h1>
<p align="center">
  <em>Real-time hand tracking and finger counting using MediaPipe and OpenCV</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success?style=for-the-badge&logo=statuspage&color=brightgreen" />
  <img src="https://img.shields.io/badge/language-Python-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/library-MediaPipe-red?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/library-OpenCV-black?style=for-the-badge&logo=opencv" />
</p>

---

## 🧠 Overview

This project is a **real-time hand tracking** and **finger counter** built with **MediaPipe Hands** and **OpenCV**. It detects up to two hands from the webcam feed and analyzes finger positions to determine how many fingers are extended.

Each detected hand is classified as **Right** or **Left**, and individual fingers (thumb, index, middle, ring, pinky) are analyzed using geometric distance and angle calculations.

---

## 📸 Screenshots

|  Five Fingers | Five Fingers | Ring finger and pinky down | Ring finger, thumb and index finger down |
|---------------|---------------|------|---------------|
| ![](./images/run1.png) | ![](./images/run2.png) | ![](./images/run3.png) | ![](./images/run4.png) |

> 📂 Make sure to place the 4 screenshots inside an `images/` folder within your project.

---

## ✨ Features

✔ Detect up to **2 hands** simultaneously  
✔ Classify **left** or **right** hand  
✔ Real-time **finger counting**  
✔ Displays:
- Total number of hands
- Finger names and which are extended
- Per-hand finger count with colored indicators  
✔ Built-in **palm centroid detection**  
✔ Works at **720p** resolution by default  
✔ Keypoint visualization with **MediaPipe drawing utils**

---

## 🛠 Tech Stack

- **Language:** Python 3.x  
- **Computer Vision:** OpenCV  
- **Hand Detection & Landmarks:** [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)  
- **Math & Geometry:** NumPy, trigonometry, angle computation  

---

## ✅ Requirements

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy
