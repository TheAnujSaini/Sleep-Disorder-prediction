Sleep Disorder Prediction Using Machine Learning
Abstract
This project presents a Machine Learning–based system for predicting sleep disorders using lifestyle and health-related parameters. The objective is to identify whether an individual is likely to experience Insomnia or Sleep Apnea based on factors such as sleep duration, stress level, physical activity, heart rate, and body mass index (BMI). The system is implemented as an interactive web application using Streamlit and serves as a decision-support tool for educational and research purposes.
________________________________________
Introduction
Sleep disorders are increasingly prevalent due to modern lifestyle factors such as stress, irregular sleep schedules, and reduced physical activity. Early detection of sleep-related issues can help individuals seek timely medical attention and improve overall well-being. This project demonstrates how Machine Learning techniques can be applied to analyze lifestyle data and predict potential sleep disorders.
________________________________________
Objectives
•	To develop a predictive model for identifying sleep disorders using Machine Learning.
•	To analyze the influence of lifestyle and physiological factors on sleep health.
•	To design a user-friendly web interface for real-time prediction.
•	To demonstrate the practical application of Machine Learning in healthcare analytics.
________________________________________
Dataset
The model is trained on the Sleep Health and Lifestyle Dataset, which contains information related to an individual's daily habits, health indicators, and sleep quality.
Input Features
•	Gender
•	Age
•	Occupation
•	Sleep Duration
•	Quality of Sleep
•	Physical Activity Level
•	Stress Level
•	BMI Category
•	Heart Rate
•	Daily Steps
•	Systolic Blood Pressure
•	Diastolic Blood Pressure
Target Variable
•	Insomnia
•	Sleep Apnea
________________________________________
Methodology
Data Preprocessing
•	Handled missing and inconsistent values.
•	Converted categorical variables using Label Encoding.
•	Standardized numerical features using StandardScaler.
•	Split dataset into training and testing sets (80:20).
Model Development
Multiple classification algorithms were evaluated, including:
•	Decision Tree Classifier
•	Random Forest Classifier
•	Support Vector Machine
The Random Forest Classifier was selected as the final model due to its superior accuracy and robustness.
Model Performance
•	Training Accuracy: Approximately 98–99%
•	Testing Accuracy: Approximately 93–95%
________________________________________
System Implementation
The trained Machine Learning model, along with preprocessing components (scaler, encoders, and feature mapping), was saved using the joblib library and integrated into a Streamlit-based web application.
The application allows users to input lifestyle and health parameters and receive real-time predictions indicating whether the individual is more likely to experience Insomnia or Sleep Apnea.
________________________________________
Deployment
The application can be deployed using:
•	Streamlit Community Cloud
•	Hugging Face Spaces
Deployment involves uploading the project repository, specifying dependencies in requirements.txt, and running the Streamlit application.
________________________________________
Requirements
streamlit
scikit-learn
joblib
numpy
pandas
________________________________________
Usage
To run the project locally:
git clone https://github.com/your-username/sleep-disorder-prediction.git
cd sleep-disorder-prediction
pip install -r requirements.txt
streamlit run app.py
________________________________________
Applications
•	Preventive healthcare and lifestyle monitoring
•	Educational demonstration of Machine Learning in healthcare
•	Wellness and sleep health awareness tools
________________________________________
Limitations
•	The dataset size is limited, which may affect generalization.
•	The system is not a substitute for clinical diagnosis.
•	Some medically relevant parameters are not included in the dataset.
________________________________________
Future Scope
•	Incorporation of larger and clinically validated datasets.
•	Multi-class prediction including normal sleep condition.
•	Integration with wearable health monitoring devices.
•	Development of a mobile application interface.
________________________________________
Conclusion
This project demonstrates the effective use of Machine Learning techniques for predicting sleep disorders using lifestyle and health-related data. The system provides a simple and interactive platform for early awareness and highlights the potential of data-driven approaches in healthcare analytics.
________________________________________
Author
Anuj Saini
Bachelor of Technology (Computer Science and Engineering)
Specialization: Data Science and Machine Learning
________________________________________
License
This project is intended for academic and research purposes only and should not be used as a medical diagnostic system.
