😴 Sleep Disorder Prediction using Machine Learning

📌 Project Overview
This project is a Machine Learning–based web application that predicts whether a person is likely to suffer from Insomnia or Sleep Apnea based on lifestyle and health parameters.
The system analyzes sleep patterns, stress level, physical activity, heart rate, BMI, and other health indicators to provide an early prediction of possible sleep disorders.

⚠️ This is an educational and research project. It is not a medical diagnosis tool.

🎯 Objectives
Predict sleep disorders using Machine Learning
Build a user-friendly web interface for real-time prediction
Analyze the impact of lifestyle and health factors on sleep
Demonstrate practical application of ML in healthcare

📊 Dataset
The model is trained on the Sleep Health and Lifestyle Dataset.
Features used:
Gender
Age
Occupation
Sleep Duration
Quality of Sleep
Physical Activity Level
Stress Level
BMI Category
Heart Rate
Daily Steps
Systolic Blood Pressure
Diastolic Blood Pressure

Target:
Insomnia
Sleep Apnea

🤖 Machine Learning Approach
Data Preprocessing
Handled missing values
Label Encoding for categorical variables
Feature scaling using StandardScaler
Train–Test split (80/20)
Models Tested
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine
Final Model: Random Forest Classifier
Accuracy: ~93–95%

💾 Model Saving
The trained model pipeline includes:
Trained Random Forest model
StandardScaler
Label Encoders
Feature column order
Saved using:
joblib.dump(pipeline, "sleep_pipeline2.pkl")

🌐 Streamlit Web Application
Features
Clean and interactive UI
Real-time prediction
Automatic encoding & scaling
Background header design
Highlighted prediction output
Confidence score display
User inputs lifestyle and health data → Model predicts Insomnia or Sleep Apnea.

🚀 Deployment
This project can be deployed on:
Streamlit Community Cloud
Hugging Face Spaces
Deployment Steps
Upload project to GitHub
Add requirements.txt
Connect repository to Streamlit Cloud / Hugging Face
Deploy and run

📦 Requirements
Create requirements.txt:
streamlit
scikit-learn
joblib
numpy
pandas

🖥️ Run Locally
git clone https://github.com/your-username/sleep-disorder-prediction.git
cd sleep-disorder-prediction
pip install -r requirements.txt
streamlit run app.py

📈 Applications
Health monitoring systems
Lifestyle and wellness apps
Preventive healthcare tools
Educational ML healthcare demo

⚠️ Limitations
Small dataset
Not a clinical diagnosis
Limited medical features

🔮 Future Improvements
Add multi-class prediction (Normal / Insomnia / Sleep Apnea)
Use larger medical dataset
Add deep learning models
Mobile app integration
Wearable sensor data support


Anuj Saini
B.Tech CSE | Data Science & Machine Learning
