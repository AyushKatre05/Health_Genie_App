import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title('Diabetes Checkup')
    
    
    data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive/diabetes.csv')
    
    
    columns_to_drop = ['Outcome', 'slope', 'ca','thal']
    x = data.drop(['Outcome'], axis=1)
    y = data['Outcome']

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    
    def user_report():
        st.selectbox('Enter',['None', 'Mild', 'Moderate', 'Severe'])
        st.multiselect('What are your favorite colors', ['Green', 'Yellow', 'Red', 'Blue'])
        pregnancies = st.slider('Pregnancies', 0, 17, 3)
        glucose = st.slider('Glucose', 0, 200, 120)
        bp = st.slider('Blood Pressure', 0, 122, 70)
        skin = st.slider('Skin Thickness', 0, 100, 20)
        insulin = st.slider('Insulin', 0, 846, 120)
        bmi = st.slider('BMI', 0.0, 67.1, 20.0)
        dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        age = st.slider('Age', 1, 88, 33)

        user_report = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    
    user_data = user_report()

    
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    
    st.subheader('Model Accuracy')
    accuracy = accuracy_score(y_test, rf.predict(x_test))
    st.write(f'Accuracy: {accuracy:.2f}')

    
    st.subheader('Your Report')
    user_result = rf.predict(user_data)
    if user_result[0] == 0:
        st.write('You are not diabetic.')
    else:
        st.write('You are diabetic.')

if __name__ == "__main__":
    main()
