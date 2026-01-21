# 📧 Email Spam Detection Project

## 📌 Description

This project focuses on detecting whether an **email message is Spam or Not Spam** using **machine learning techniques**.  
The prediction is based on the **text content of email messages** using **Natural Language Processing (NLP)**.

The project is implemented using a **Jupyter Notebook (.ipynb)** and includes **data preprocessing**, **text vectorization**, **model training**, **evaluation**, and **spam prediction**.

---

## 🎯 Objectives

* Analyze email text dataset
* Perform text cleaning and preprocessing
* Convert text data into numerical features using TF-IDF
* Train a machine learning classification model
* Predict whether an email is Spam or Not Spam

---

## 📂 Project Structure

Email_Spam_Detection/
│
├── notebook/
│ └── Email_Spam_Detection.ipynb # Jupyter Notebook
│
├── data/
│ └── spam.csv # Dataset
│
├── outputs/
│ ├── spam_vs_non_spam.png
│ └── confusion_matrix.png
│
├── requirements.txt # Dependencies
└── README.md

---

## 🛠 Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Natural Language Processing (NLP)
* Google Colab / Jupyter Notebook

---

## 📊 Sample Output

### 1️⃣ Spam vs Non-Spam Distribution

This graph shows the count of **Spam** and **Not Spam** email messages.

![Spam vs Non-Spam](outputs/spam_vs_non_spam.png)

---

### 2️⃣ Confusion Matrix

This visualization represents the performance of the spam detection model.

![Confusion Matrix](outputs/confusion_matrix.png)

---

### 3️⃣ Model Evaluation

The model performance is evaluated using:
* **Accuracy Score**
* **Classification Report**
* **Confusion Matrix**

These metrics are printed directly in the notebook.

---

### 4️⃣ Custom Email Prediction

The model can also classify a **custom email message** entered by the user.



## ✅ Conclusion

The machine learning model successfully classifies emails as **Spam** or **Not Spam** with high accuracy.  
This project demonstrates the practical application of **NLP** and **classification algorithms** in real-world email filtering systems.

---

## 👩‍💻 Author

**Rupali Sutar**  
BE Computer Engineering  
Data Science & Machine Learning Enthusiast

