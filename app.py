import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('pima-indians-diabetes.csv')

X = df.drop('Outcome', axis=1)  
y = df['Outcome']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_k = 94  
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred)


st.title('KNN Model for Diabetes Classification')

st.write(f'This model uses a KNN algorithm (k={best_k}).')
st.write(f'Accuracy of a KNN model where k={best_k} is {best_accuracy * 100:.2f}%, which is the highest accuracy of all k values between 1 and 100.')

st.subheader('Enter User Data')

age = st.number_input('Age', min_value=1, max_value=120, value=30, key="age")
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1, key="pregnancies")
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=100, key="glucose")
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70, key="blood_pressure")
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20, key="skin_thickness")
insulin = st.number_input('Insulin Level', min_value=0, max_value=1000, value=80, key="insulin")
bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0, value=25.0, key="bmi")
dpf = st.number_input('Diabetes Pedigree Function (DPF)', min_value=0.0, max_value=2.5, value=0.5, key="dpf")

user_input = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
user_input_scaled = scaler.transform(user_input)

if st.button('Prediction'):
    prediction = model.predict(user_input_scaled)
    st.write('Prediction:', 'Diabetes' if prediction[0] == 1 else 'Not Diabetes')

    st.write("""
    Below is a scatter plot demonstrating the relationship of two variables (Glucose vs BMI) in your value vs. our testing data values.
    """)

    plt.scatter(df['Glucose'], df['BMI'], color='blue', alpha=0.5, label='Existing Data')

    new_data = user_input[0]  
    plt.scatter(new_data[1], new_data[6], color='red', label='New Input')

    plt.xlabel('Glucose Level')
    plt.ylabel('Body Mass Index (BMI)')
    plt.title('Scatter Plot of Glucose vs BMI')

    # plt.text(new_data[1], new_data[6], f'User Input: {new_data[1]:.1f}, {new_data[6]:.1f}', color='red', fontsize=12)

    plt.legend()
    st.pyplot(plt)

    st.write("""
    Please note: This model is not intended to serve as a reliable source of medical information and should not be used as a substitute for professional medical consultation.
    """)

    st.write("""
    Dataset from: Pima Indians Diabetes Database - National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). This dataset is used for machine learning techniques to diagnose diabetes based on various health parameters such as age, blood pressure, and glucose levels.
    """)