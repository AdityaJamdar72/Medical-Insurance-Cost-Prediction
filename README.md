# Medical-Insurance-Cost-Prediction
📌Project Overview :
This project aims to predict medical insurance charges based on individual attributes such as age, BMI, smoking status, and number of children.
The goal is to build a regression model that estimates insurance costs and deploy it using a Streamlit web application.


📊 Dataset
Source: Public medical insurance dataset (Kaggle)
Target Variable: charges
Number of Features: 6 (after preprocessing)

🧹 Data Preprocessing
Checked missing values (dataset contains no missing values)
Encoded categorical variables (sex, smoker)
Removed region due to low predictive impact
Applied feature scaling using StandardScaler
Split data into training and testing sets

📈 Model Performance
Model	R² Score
Linear Regression	~0.78
Ridge Regression	~0.78
Lasso Regression	~0.78
MAE and MSE values are within acceptable ranges
Linear models occasionally produce negative predictions, which were clipped to zero since insurance charges cannot be negative

🖥️ Streamlit Web App
A Streamlit application was built to:
Take user input (age, BMI, children, sex, smoker)
Validate input ranges
Display predicted insurance charges interactively
Key UI Features
Number input constraints to avoid unrealistic values
Predict button to prevent auto-prediction

# to run this code us 'python -m streamlit run Medcost.py'

🧑‍💻 Author
Aditya , Machine Learning Enthusiast
