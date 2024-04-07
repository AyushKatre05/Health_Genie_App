import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu

def main():
    st.title("Heart Disease Checkup - Health Genie")


    
    data = pd.read_csv(
        "C:/Users/ASUS Vivobook/Downloads/archive (4)/Heart_Disease_Prediction.csv"
    )

    
    x = data.drop("Heart Disease", axis=1)
    y = data["Heart Disease"]

    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    
    def user_report():
        age = st.slider("Patient Age", 20, 90, 40)
        sex = st.radio("Patient Sex", ["Male", "Female"])
        cp = st.slider("Chest Pain Type", 0, 3, 0)
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.slider("Serum Cholesterol", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar", ["< 120 mg/dl", "> 120 mg/dl"])
        restecg = st.slider("Resting ECG Results", 0, 2, 0)
        max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
        st_depression = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 0.0)
        slope_of_ST = st.slider("Slope of ST", 90, 200, 120, key="slope_of_ST")
        num_of_vessels_fluro = st.slider(
            "Number of Vessels Fluro", 0, 3, 0, key="num_of_vessels_fluro"
        )
        thallium = st.slider("Thallium", 0, 3, 0, key="thallium")

        sex_encoded = 1 if sex == "Male" else 0
        fbs_encoded = 1 if fbs == "> 120 mg/dl" else 0
        exang_encoded = 1 if exang == "Yes" else 0

        user_report = {
            "Age": age,
            "Sex": sex_encoded,
            "Chest pain type": cp,
            "BP": trestbps,
            "Cholesterol": chol,
            "FBS over 120": fbs_encoded,
            "EKG results": restecg,
            "Max HR": max_hr,
            "Exercise angina": exang_encoded,
            "ST depression": st_depression,
            "Slope of ST": slope_of_ST,
            "Number of vessels fluro": num_of_vessels_fluro,
            "Thallium": thallium,
        }
        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

   
    user_data = user_report()

    
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    
    st.subheader("Your Report")
    user_result = rf.predict(user_data)
    if user_result[0] == "Presence":
        st.write("You are facing heart disease.")
    else:
        st.write("You are safe from heart disease.")


if __name__ == "__main__":
    main()
