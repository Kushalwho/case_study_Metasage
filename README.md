🚀 Project Overview


This project analyzes aftermarket car modifications using COBB Dyno Data to predict how changes impact performance. It utilizes:

Machine Learning (Linear Regression, TF-IDF, Scikit-learn)
Data Preprocessing (Scaling, TF-IDF for text-based features)
Streamlit Web App (User input & predictions)

📂 Dataset
The dataset contains dynamometer (dyno) test results and car modification details. Key features:

RPM (Revolutions per Minute)
Boost Pressure (PSI)
Torque (Nm)
Air-Fuel Ratio (AFR)
Modifications (Text data) 

💻 Installation & Setup
1. Prerequisites
   Ensure you have:
   Python 3.7+
   pip (Python package manager)

2. Download the Dataset (available in zip file)
   Ensure merged_dyno_car_info.csv is available in the project directory.

▶️ Running the App
   Start the Streamlit web app by running:
   streamlit run app.py

🛠️ Technical Details
1️. Data Ingestion
   Stored as CSV (Flat File)
   Loaded using Pandas
   Processed using NumPy & Scikit-learn
   
2️. Preprocessing
   Numerical Features: Scaled with StandardScaler
   Text Features: Processed with TfidfVectorizer (max 202 features)
   
3️. Model Details
   Linear Regression Model
   Features Used:
   Torque, RPM, AFR, Boost
   TF-IDF Encoded Specs
   
4️. Model Deployment
   Uses Streamlit to provide a simple, interactive interface.
   Models & vectorizers are loaded from .pkl files.   
 

