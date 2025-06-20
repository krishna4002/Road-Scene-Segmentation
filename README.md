# 🛣 Road Scene Segmentation for Autonomous Driving

This project is a smart AI tool that *automatically identifies different objects and elements on the road, such as vehicles, pedestrians, lanes, potholes, sidewalks, traffic signs, and more. It can work with images and even in **real time using your webcam, making it suitable for **self-driving car research, **smart city applications, and **road safety monitoring*.

---

## 🔍 What This Project Does (For Everyone)

- 🧠 It uses *AI to understand road scenes* from camera input.
- 🚘 It was trained on *real driving data from Indian roads* and *pothole images*.
- 🎥 It works in real-time with a *webcam* or with *uploaded road images*.
- 🎨 It shows you different parts of the road in *colored labels* so you can understand what the AI sees.

---

## 🧠 Behind the Scenes (For Technical Users)

This project uses:
- ✅ A *DeepLabV3 semantic segmentation model*
- ✅ Trained on *IDD (India Driving Dataset)* + *Roboflow pothole dataset*
- ✅ Supports 28 classes: 27 from IDD, 1 pothole
- ✅ Written in *Python* using *PyTorch* and *Streamlit* for the UI

---

## 🚀 Features

- 🛣 Recognizes and segments road, lanes, vehicles, buildings, sky, people, potholes, etc.
- 📸 Works with uploaded images
- 🎥 Works live with a webcam stream
- 🖍 Displays *color-coded masks* with a *legend* for 28 classes
- ✅ Easy to use – no coding knowledge required to run the app

---

## 📁 Project Structure


Road Scene Segmentation for Autonomous Driving/
├── app.py                       # Main Streamlit UI
├── realtime.py                 # Optional: Opens webcam in separate window
├── utils.py                    # Color map & helper functions
├── preprocess.py               # Merges IDD + Pothole dataset
├── deeplabv3.pth               # Trained model
├── data/
│   ├── idd_segmentation/       # Manually downloaded dataset
│   └── pothole-detection/      # Roboflow dataset (COCO Seg format)
├── requirements.txt            # Required Python packages
└── README.md


---

## 🛠 How to Use (Simple Steps)

### Step 1: Install Everything

#### Option A: Easy way (recommended for beginners)
bash
python -m venv venv
venv\Scripts\activate          # For Windows
# or
source venv/bin/activate       # For Linux/macOS

pip install -r requirements.txt


### Step 2: Download the Datasets

#### ✅ IDD Dataset (Manual Step)
- Go to: https://idd.insaan.iiit.ac.in/
- Download: *IDD Segmentation Part I*
- Extract it inside data/idd_segmentation/

#### ✅ Roboflow Pothole Dataset (Automatic)
Already integrated into the app using Roboflow API:
python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("jay-ngdms").project("pothole-detection-system-rnoh4-xjw6w")
version = project.version(1)
version.download("coco-segmentation")


### Step 3: Preprocess the Dataset
bash
python preprocess.py


### Step 4: Run the App
bash
streamlit run app.py


---

## 🎮 Two App Modes

### 📸 Image Segmentation Mode
- Upload a photo of a road
- AI will highlight roads, cars, people, sky, potholes, etc.
- Colored result + class legend displayed

### 🎥 Real-Time Segmentation Mode
- Turn on your webcam
- AI will segment and label objects *live*
- Helps simulate how self-driving cars "see" the road

---

## 🧾 Class Legend (28 Classes)

The app shows a *legend* (key) for each class and its color.
| ID | Class Name     |
|----|----------------|
| 0  | Road           |
| 1  | Sidewalk       |
| 2  | Building       |
| 3  | Wall           |
| 4  | Fence          |
| 5  | Pole           |
| 6  | Traffic Light  |
| 7  | Traffic Sign   |
| 8  | Vegetation     |
| 9  | Terrain        |
| 10 | Sky            |
| 11 | Person         |
| 12 | Rider          |
| 13 | Car            |
| 14 | Truck          |
| 15 | Bus            |
| 16 | Train          |
| 17 | Motorcycle     |
| 18 | Bicycle        |
| 19 | Lane Marking   |
| 20 | Parking Area   |
| 21 | Animal         |
| 22 | Debris         |
| 23 | Manhole        |
| 24 | Water Body     |
| 25 | Billboard      |
| 26 | Bridge         |
| 27 | Pothole        |

Each is color-coded using *visually distinct and non-similar colors* for clarity.

---

## 🧠 How the Model Works

- Based on *DeepLabV3 with ResNet-50*
- Re-trained with custom final layer: 28 output channels
- Trained from scratch on merged dataset using PyTorch
- Works on CPU or GPU

---

## 👀 Example Results

> (Add screenshots or demo GIFs here showing before and after segmentation on real roads and webcam)

---

## 📄 License

This project is licensed under the *MIT License* — you are free to use, modify, and distribute it.

---

## 🙏 Acknowledgements

- [IDD Dataset - IIIT Hyderabad](https://idd.insaan.iiit.ac.in/)
- [Roboflow Pothole Dataset](https://roboflow.com/)
- [PyTorch DeepLabV3](https://pytorch.org/vision/stable/models.html#deeplabv3)
- [Streamlit](https://streamlit.io/)

---

## 📬 Feedback or Contributions

💡 Ideas, bug reports, or contributions? Open an issue or PR on GitHub!  
🔗 GitHub: https://github.com/YOUR_USERNAME/road-scene-segmentation
