import numpy as np
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Fungsi untuk melatih model dan melakukan prediksi
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Buat model Logistic Regression
    model = LogisticRegression(max_iter=200)
    
    # Latih model menggunakan data training
    model.fit(X_train, y_train)
    
    # Simpan model dengan joblib
    # joblib.dump(model, MODEL_PATH)
    # Simpan model dengan joblib
    try:
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved at {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")


    print(f"Model saved at {MODEL_PATH}")
    
    # Prediksi menggunakan data testing
    prediction = model.predict(X_test)
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, prediction)
    
    # Buat confusion matrix
    conf_matrix = confusion_matrix(y_test, prediction)
    
    return prediction, accuracy, conf_matrix

# Load dataset Iris
iris = load_iris()
X = iris.data  # Fitur
y = iris.target  # Label

# Directory to save the model
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "log_regress_model.joblib")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Panggil fungsi train_and_evaluate
prediction, accuracy, conf_matrix = train_and_evaluate(X_train, X_test, y_train, y_test)

# Cetak hasil prediksi, akurasi, dan confusion matrix
print(f"Akurasi: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)