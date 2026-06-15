# 🩺 Breast Cancer Classification Using Neural Networks

## 📌 Project Overview

This project uses a Deep Learning Neural Network built with TensorFlow/Keras to classify breast cancer tumors as **Benign** or **Malignant** based on medical diagnostic features from the Breast Cancer Wisconsin Dataset.

The application provides a user-friendly web interface built with Streamlit, allowing users to enter tumor characteristics and receive instant predictions.

---

## 🚀 Features

* Deep Learning model using TensorFlow/Keras
* Interactive Streamlit web application
* Real-time prediction of tumor type
* Data preprocessing using Scikit-Learn
* Model and scaler persistence using Pickle and Keras

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* Streamlit
* Joblib / Pickle

---

## 📂 Project Structure

```text
Breast_Cancer_Prediction/
│
├── app.py
├── best_model.keras
├── scaler.pkl
├── requirements.txt
├── README.md
└── DL_Project_1_Breast_Cancer_Classification_with_NN.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Breast_Cancer_Prediction.git
cd Breast_Cancer_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
https://huggingface.co/spaces/TEJASMAURYA/breast-cancer-prediction
```

---

## 📊 Input Features

The model uses 30 diagnostic features including:

* Mean Radius
* Mean Texture
* Mean Perimeter
* Mean Area
* Mean Smoothness
* Mean Compactness
* Mean Concavity
* Mean Symmetry
* Worst Radius
* Worst Texture
* Worst Perimeter
* Worst Area and more

and other related tumor measurements.

---

## 🎯 Prediction Output

The model predicts:

* 🟢 Benign (Non-Cancerous)
* 🔴 Malignant (Cancerous)

along with a prediction confidence score.

---

## 📈 Future Improvements

* Feature importance visualization
* Model explainability with SHAP
* Enhanced UI/UX
* Cloud deployment support
* Additional medical analytics

---

## 👨‍💻 Author

Tejansh Maurya

Machine Learning & Deep Learning Enthusiast
