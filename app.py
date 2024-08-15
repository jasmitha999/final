import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import cv2
import statistics
import pymysql
import base64
from fpdf import FPDF
from datetime import datetime
import os

# Function to connect to the database
def connect_db():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

# Function to create a new user account
def create_account(name, email, number, age, gender, password):
    try:
        connection = connect_db()
        with connection.cursor() as cursor:
            query = "INSERT INTO users (name, email, number, age, gender, password) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (name, email, number, age, gender, password))
            connection.commit()
            st.success("Account created successfully!")
    except Exception as e:
        st.error(f"Error creating account: {e}")
    finally:
        connection.close()

# Function to check user login credentials
def check_login(email, password):
    try:
        connection = connect_db()
        with connection.cursor() as cursor:
            query = "SELECT * FROM users WHERE email=%s AND password=%s"
            cursor.execute(query, (email, password))
            user = cursor.fetchone()
            if user:
                st.success("Login successful!")
                return user
            else:
                st.error("Invalid credentials.")
    except Exception as e:
        st.error(f"Error logging in: {e}")
    finally:
        connection.close()

# Function to check if an image is an ECG image with valid ECG waves
def is_ecg_image(image_path):
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Heuristic to detect ECG-like wave patterns based on contours and their distribution
    valid_ecg_wave_contours = [contour for contour in contours if 10 < cv2.boundingRect(contour)[3] < 100]
    
    # Set a threshold for the number of valid contours to consider the image as an ECG image
    return len(valid_ecg_wave_contours) > 50  # Adjust the threshold as needed

# Function to extract metadata from an ECG image
def extract_ecg_metadata(image_path):
    ecg_image = cv2.imread(image_path)
    resized_image = cv2.resize(ecg_image, (2213, 1572))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(binary_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    st_elevation_values = []
    pathological_q_waves_values = []
    t_wave_inversions_values = []
    abnormal_qrs_complexes_values = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20:
            st_elevation_values.append(h)
        if w > 10 and h < 5:
            pathological_q_waves_values.append(h)
        if h < 10 and w > 15:
            abnormal_qrs_complexes_values.append(h)
        if h < 10 and w < 10:
            t_wave_inversions_values.append(h)

    def calculate_stats(values):
        return {
            'max': max(values) if values else 0,
            'mean': statistics.mean(values) if values else 0,
            'median': statistics.median(values) if values else 0
        }

    metadata = {
        'Max ST Elevation (Height)': calculate_stats(st_elevation_values)['max'],
        'Mean ST Elevation (Height)': calculate_stats(st_elevation_values)['mean'],
        'Median ST Elevation (Height)': calculate_stats(st_elevation_values)['median'],
        'Max Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['max'],
        'Mean Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['mean'],
        'Median Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['median'],
        'Max T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['max'],
        'Mean T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['mean'],
        'Median T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['median'],
        'Max Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['max'],
        'Mean Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['mean'],
        'Median Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['median']
    }

    return metadata

# Function to predict disease based on ECG metadata
def predict_disease(image_path, model, scaler, label_encoder):
    if not is_ecg_image(image_path):
        return "Please upload a valid ECG image."
    
    metadata = extract_ecg_metadata(image_path)
    metadata_df = pd.DataFrame([metadata])

    if scaler:
        metadata_scaled = scaler.transform(metadata_df)
        prediction_index = model.predict(metadata_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction_index])[0]
        return predicted_class
    else:
        return "Error: Scaler not available."

# Function to get precautions based on predicted disease type
def get_precautions(disease_type):
    precautions = {
        "myocardial": [
            "1. Take prescribed medications as directed.",
            "2. Avoid heavy physical exertion.",
            "3. Monitor heart rate and report any irregularities.",
            "4. Follow up with a cardiologist regularly."
        ],
        "historyofmi": [
            "1. Maintain a healthy lifestyle with a balanced diet.",
            "2. Exercise regularly but avoid strenuous activities.",
            "3. Regular check-ups with a healthcare provider.",
            "4. Keep track of any new symptoms."
        ],
        "abnormal": [
            "1. Follow up with a healthcare provider for further evaluation.",
            "2. Monitor for any changes in symptoms.",
            "3. Maintain a healthy lifestyle."
        ],
        "normal": [
            "1. Maintain a healthy diet rich in fruits, vegetables, and whole grains.",
            "2. Engage in regular physical activity, such as walking or cycling.",
            "3. Avoid smoking and limit alcohol intake.",
            "4. Get regular health check-ups and monitor blood pressure regularly."
        ]
    }
    return precautions.get(disease_type.lower(), ["No specific precautions available."])

# Function to generate a PDF report
def generate_pdf_report(user_name, predicted_class, metadata, precautions):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {user_name}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Disease Type: {predicted_class}", ln=True)
    
    pdf.cell(200, 10, txt="Metadata:", ln=True)
    for key, value in metadata.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.cell(200, 10, txt="Precautions:", ln=True)
    for precaution in precautions:
        pdf.cell(200, 10, txt=precaution, ln=True)
    
    pdf_output = f"{user_name}_heart_disease_report.pdf"
    pdf.output(pdf_output)
    
    return pdf_output

# Load pre-trained models and other assets
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
cat_model = joblib.load('cat_model.joblib')
voting_clf = joblib.load('voting_clf.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Streamlit app setup
def main():
    st.title("Heart Disease Prediction App")
    
    menu = ["Home", "Login", "Sign Up", "Predict", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the Heart Disease Prediction App")
    
    elif choice == "Sign Up":
        st.subheader("Create a New Account")
        
        name = st.text_input("Name")
        email = st.text_input("Email")
        number = st.text_input("Phone Number")
        age = st.number_input("Age", min_value=1, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        password = st.text_input("Password", type="password")
        
        if st.button("Create Account"):
            create_account(name, email, number, age, gender, password)
    
    elif choice == "Login":
        st.subheader("Login to Your Account")
        
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user = check_login(email, password)
    
    elif choice == "Predict":
        st.subheader("Upload ECG Image for Prediction")
        
        user_name = st.text_input("Enter your name")
        image_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
        
        if st.button("Predict"):
            if image_file is not None:
                # Save the uploaded image file
                image_path = f"./temp_{datetime.now().timestamp()}.png"
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
                
                # Make a prediction
                predicted_class = predict_disease(image_path, voting_clf, scaler, label_encoder)
                st.write(f"Predicted Disease: {predicted_class}")
                
                # Get metadata and precautions
                metadata = extract_ecg_metadata(image_path)
                precautions = get_precautions(predicted_class)
                
                st.write("Precautions:")
                for precaution in precautions:
                    st.write(precaution)
                
                # Generate and provide a PDF report
                pdf_report = generate_pdf_report(user_name, predicted_class, metadata, precautions)
                with open(pdf_report, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    b64 = base64.b64encode(pdf_bytes).decode()
                    download_link = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_report}">Download Report</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
    
    elif choice == "About":
        st.subheader("About the Heart Disease Prediction App")
        st.write("""
            This app uses machine learning to predict the likelihood of heart disease based on ECG images. 
            It utilizes multiple models, including RandomForest, XGBoost, CatBoost, and a Voting Classifier. 
            It also generates personalized precautions and provides a downloadable PDF report.
        """)

if __name__ == "_main_":
    main()