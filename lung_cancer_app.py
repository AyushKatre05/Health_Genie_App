import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    st.title('Health Genie - Lung Cancer Prediction')

    
    data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (6)/survey lung cancer.csv')

    
    x = data.drop(['LUNG_CANCER'], axis=1)
    y = data['LUNG_CANCER']
    
    
    le = LabelEncoder()
    x['GENDER'] = le.fit_transform(x['GENDER'])

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    
    def user_report():
        gender = st.sidebar.selectbox('Gender', ['M', 'F'], index=1)  # Default value set to 'F'
        age = st.sidebar.slider('Age', 1, 100, 50)
        smoking = st.sidebar.selectbox('Smoking', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        yellow_fingers = st.sidebar.selectbox('Yellow Fingers', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        anxiety = st.sidebar.selectbox('Anxiety', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        peer_pressure = st.sidebar.selectbox('Peer Pressure', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        chronic_disease = st.sidebar.selectbox('Chronic Disease', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        fatigue = st.sidebar.selectbox('Fatigue', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        allergy = st.sidebar.selectbox('Allergy', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        wheezing = st.sidebar.selectbox('Wheezing', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        alcohol = st.sidebar.selectbox('Alcohol', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        coughing = st.sidebar.selectbox('Coughing', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        shortness_of_breath = st.sidebar.selectbox('Shortness of Breath', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        swallowing_difficulty = st.sidebar.selectbox('Swallowing Difficulty', ['YES', 'NO'], index=1)  # Default value set to 'NO'
        chest_pain = st.sidebar.selectbox('Chest Pain', ['YES', 'NO'], index=1)  # Default value set to 'NO'

        user_report = {
            'GENDER': 0 if gender == 'M' else 1,  # Encoding 'M' as 0 and 'F' as 1
            'AGE': age,
            'SMOKING': 2 if smoking == 'YES' else 1,
            'YELLOW_FINGERS': 2 if yellow_fingers == 'YES' else 1,
            'ANXIETY': 2 if anxiety == 'YES' else 1,
            'PEER_PRESSURE': 2 if peer_pressure == 'YES' else 1,
            'CHRONIC DISEASE': 2 if chronic_disease == 'YES' else 1,
            'FATIGUE ': 2 if fatigue == 'YES' else 1,
            'ALLERGY ': 2 if allergy == 'YES' else 1,
            'WHEEZING': 2 if wheezing == 'YES' else 1,
            'ALCOHOL CONSUMING': 2 if alcohol == 'YES' else 1,
            'COUGHING': 2 if coughing == 'YES' else 1,
            'SHORTNESS OF BREATH': 2 if shortness_of_breath == 'YES' else 1,
            'SWALLOWING DIFFICULTY': 2 if swallowing_difficulty == 'YES' else 1,
            'CHEST PAIN': 2 if chest_pain == 'YES' else 1
        }
        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    
    user_data = user_report()

    
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    

    
    st.subheader('Your Report')
    if st.sidebar.button('PREDICT'):
        user_result = rf.predict(user_data)
        if user_result[0] == 'YES':
            st.write('__You may have lung cancer. Please consult a doctor.__')
        else:
            st.write('__You are unlikely to have lung cancer.__')
        
        st.markdown("""
        *Disclaimer:* This is a machine learning-based model for disease prediction. While it can provide insights, it may not always be accurate. For accurate medical advice, please consult a qualified healthcare professional.
        """)
        
if __name__ == "__main__":
    main()
