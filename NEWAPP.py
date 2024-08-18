import os
import re
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import hashlib
import sqlite3
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# Set page configuration
st.set_page_config(page_title="Health Assistant",
                layout="wide",
                page_icon="🧑‍⚕️")

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(
    open('C:/Users/anike/OneDrive/Desktop/MLP/trained_model.sav', 'rb'))

heart_disease_model = pickle.load(
    open('C:/Users/anike/OneDrive/Desktop/MLP/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(
    open('C:/Users/anike/OneDrive/Desktop/MLP/parkinsons_model.sav', 'rb'))

# Load your data
df = pd.read_csv("Training.csv")

# Map prognosis to integer values
prognosis_map = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
    'Migraine': 11, 'Cervical spondylosis': 12,
    'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
    'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31,
    'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39,
    'Impetigo': 40
}

# Define symptom options
symptom_options = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
                   'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
                   'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
                   'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
                   'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
                   'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
                   'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
                   'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
                   'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                   'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
                   'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
                   'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
                   'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                   'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
                   'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
                   'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
                   'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
                   'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
                   'yellow_crust_ooze']


df['prognosis'] = df['prognosis'].map(prognosis_map)

# Fill missing values in prognosis column with the most frequent value
df['prognosis'] = df['prognosis'].fillna(df['prognosis'].mode()[0])

# Fill missing values in numeric columns with the mean
for col in symptom_options:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())


# Function to connect to the SQLite database
def connect_db():
    conn = sqlite3.connect('userdata.db')
    return conn

# Cursor is used to execute SQL queries on the database
# Function to create tables if they don't exist


def create_tables():
    conn = connect_db()  # Connect to the database
    cursor = conn.cursor()  # Create a cursor

    # Create user table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    # Create user input table with user_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_input_diabetes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            Pregnancies INTEGER,
            Glucose INTEGER,
            BloodPressure INTEGER,
            SkinThickness INTEGER,
            Insulin INTEGER,
            BMI REAL,
            DiabetesPedigreeFunction REAL,
            Age INTEGER,
            Diagnosis TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # for heart disease
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_input_heart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            Diagnosis TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # for parkinsons
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_input_parkinsons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            fo INTEGER,
            fhi INTEGER,
            flo INTEGER,
            jitter_present INTEGER,
            jitter_abs INTEGER,
            rap  INTEGER,
            ppq INTEGER,
            ddp INTEGER,
            shimmer INTEGER,
            shimmer_db INTEGER,
            apq3 INTEGER,
            apq5 INTEGER,
            apq INTEGER,
            dda INTEGER,
            nhr INTEGER,
            hnr INTEGER,
            rpde INTEGER,
            dfa INTEGER,
            spread1 INTEGER,
            spread2 INTEGER,
            d2 INTEGER,
            ppe INTEGER,
            Diagnosis TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')


    # Create disease prediction table with user_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_prediction (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symptoms TEXT,
        diagnosis TEXT,
        model_used TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to register a new user
def register_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                (username, hash_password(password)))
    conn.commit()
    conn.close()

# Function to check if a user exists


def user_exists(username):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user is not None


# Function to verify user credentials
def verify_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user is not None


# Function to insert user data into the database for diabetes
def insert_user_data_diabetes(user_id, data):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_input_diabetes (user_id, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Diagnosis)VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, *data))
    conn.commit()
    conn.close()


# Function to insert user data into the database for heart disease
def insert_user_data_heart(user_id, data):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_input_heart (user_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, Diagnosis)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, *data))
    conn.commit()
    conn.close()

# Function to insert user data into the database for parkinsons


def insert_user_data_parkinsons(user_id, data):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
                    INSERT INTO user_input_parkinsons (user_id, fo, fhi, flo, jitter_present, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe, Diagnosis)
                    VALUES ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (user_id, *data))
    conn.commit()
    conn.close()


# insert data for disease prediction
def insert_disease_data(user_id, symptoms, diagnosis, model_used):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO disease_prediction (user_id, symptoms, diagnosis, model_used)VALUES (?, ?, ?, ?)''',
        (user_id, symptoms, diagnosis, model_used))
    conn.commit()
    conn.close()

# Function to retrieve all user data
def retrieve_user_data(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_input WHERE user_id = ?', (user_id,))
    data = cursor.fetchall()
    conn.close()
    return data


# Function for prediction
def disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = diabetes_model.predict(input_data_reshaped)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'


# Function for heart disease prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = heart_disease_model.predict(input_data_reshaped)
    return 'The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease'


# Function for Parkinsons disease prediction
def parkinsons_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = parkinsons_model.predict(input_data_reshaped)
    return 'The person has Parkinsons disease' if prediction[0] == 1 else 'The person does not have Parkinsons disease'


def disease_prediction_symp(input_vector):

    # Map predictions back to disease labels
    prognosis_map_inverse = {v: k for k, v in prognosis_map.items()}
    # Train models
    clf_dt = tree.DecisionTreeClassifier()
    clf_dt.fit(df[symptom_options], df['prognosis'])

    clf_rf = RandomForestClassifier()
    clf_rf.fit(df[symptom_options], df['prognosis'])

    clf_nb = GaussianNB()
    clf_nb.fit(df[symptom_options], df['prognosis'])

    # Make predictions
    prediction_dt = clf_dt.predict(input_vector)
    prediction_rf = clf_rf.predict(input_vector)
    prediction_nb = clf_nb.predict(input_vector)

    # Map predictions back to disease labels
    prognosis_map_inverse = {v: k for k, v in prognosis_map.items()}
    diagnosis_dt = prognosis_map_inverse[prediction_dt[0]]
    diagnosis_rf = prognosis_map_inverse[prediction_rf[0]]
    diagnosis_nb = prognosis_map_inverse[prediction_nb[0]]

    return diagnosis_dt, diagnosis_rf, diagnosis_nb


# Main function
def main():
    # Create the database tables
    create_tables()

    # Create session state variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        username = st.session_state['username']
        user_id = st.session_state['user_id']

        # Sidebar with width set to 300
        with st.sidebar:
            st.write(f'Welcome, {username}!')

            # Create a spacer to push the logout button to the bottom
            st.write("")  # This creates an empty space

            # Add a button for the disease prediction options
            selected = option_menu('Multiple Disease Prediction System',
                                ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Parkinsons Prediction', 'Disease Prediction'],
                                menu_icon='hospital-fill',
                                icons=['activity', 'heart', 'person', 'heart-pulse'],
                                default_index=0)

            # Add a spacer to push the logout button down
            st.write("")  # This creates an empty space

            # Logout button at the bottom
            if st.button('Logout'):
                st.session_state['logged_in'] = False

        if selected == 'Diabetes Prediction':
            st.title('Diabetes Prediction')

            # Getting the input data from the user
            with st.form('input_form'):
                Pregnancies = st.text_input('Number of Pregnancies')
                Glucose = st.text_input('Glucose Level')
                BloodPressure = st.text_input('Blood Pressure value')
                SkinThickness = st.text_input('Skin Thickness value')
                Insulin = st.text_input('Insulin Level')
                BMI = st.text_input('BMI value')
                DiabetesPedigreeFunction = st.text_input(
                    'Diabetes Pedigree Function value')
                Age = st.text_input('Age of the Person')

                # Code for Prediction
                if st.form_submit_button('Test Disease'):
                    try:
                        input_data = [
                            int(Pregnancies),
                            int(Glucose),
                            int(BloodPressure),
                            int(SkinThickness),
                            int(Insulin),
                            float(BMI),
                            float(DiabetesPedigreeFunction),
                            int(Age)
                        ]
                        diagnosis = disease_prediction(input_data)

                        # Insert user data into the database
                        insert_user_data_diabetes(user_id, [
                            int(Pregnancies),
                            int(Glucose),
                            int(BloodPressure),
                            int(SkinThickness),
                            int(Insulin),
                            float(BMI),
                            float(DiabetesPedigreeFunction),
                            int(Age),
                            diagnosis
                        ])

                        st.success('Diagnosis: ' + diagnosis)
                    except ValueError:
                        st.error('Please enter valid input values.')

        # Heart Disease Prediction Page
        if selected == 'Heart Disease Prediction':
            st.title('Heart Disease Prediction')

            # Getting the input data from the user
            with st.form('input_form'):
                age = st.text_input('Age')
                sex = st.text_input('Sex')
                cp = st.text_input('Chest Pain types')
                trestbps = st.text_input('Resting Blood Pressure')
                chol = st.text_input('Serum Cholestoral in mg/dl')
                fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
                restecg = st.text_input('Resting Electrocardiographic results')
                thalach = st.text_input('Maximum Heart Rate achieved')
                exang = st.text_input('Exercise Induced Angina')
                oldpeak = st.text_input('ST depression induced by exercise')
                slope = st.text_input('Slope of the peak exercise ST segment')
                ca = st.text_input('Major vessels colored by flourosopy')
                thal = st.text_input(
                    'thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

                # Code for Prediction
                if st.form_submit_button('Test Disease'):
                    try:
                        input_data = [
                            int(age),
                            int(sex),
                            int(cp),
                            int(trestbps),
                            int(chol),
                            int(fbs),
                            int(restecg),
                            int(thalach),
                            int(exang),
                            float(oldpeak),
                            int(slope),
                            int(ca),
                            int(thal)
                        ]
                        Diagnosis = heart_disease_prediction(input_data)

                        # Insert user data into the database
                        insert_user_data_heart(user_id, [
                            int(age),
                            int(sex),
                            int(cp),
                            int(trestbps),
                            int(chol),
                            int(fbs),
                            int(restecg),
                            int(thalach),
                            int(exang),
                            float(oldpeak),
                            int(slope),
                            int(ca),
                            int(thal),
                            Diagnosis
                        ])
                        st.success('Diagnosis: ' + Diagnosis)
                    except ValueError:
                        st.error('Please enter valid input values.')

        # Parkinson's Prediction Page
        if selected == "Parkinsons Prediction":
            st.title("Parkinson's Disease Prediction")

            # Getting the input data from the user
            with st.form('input_form'):
                fo = st.text_input('MDVP:Fo(Hz)')
                fhi = st.text_input('MDVP:Fhi(Hz)')
                flo = st.text_input('MDVP:Flo(Hz)')
                jitter_percent = st.text_input('MDVP:Jitter(%)')
                jitter_abs = st.text_input('MDVP:Jitter(Abs)')
                rap = st.text_input('MDVP:RAP')
                ppq = st.text_input('MDVP:PPQ')
                ddp = st.text_input('Jitter:DDP')
                shimmer = st.text_input('MDVP:Shimmer')
                shimmer_db = st.text_input('MDVP:Shimmer(dB)')
                apq3 = st.text_input('Shimmer:APQ3')
                apq5 = st.text_input('Shimmer:APQ5')
                apq = st.text_input('MDVP:APQ')
                dda = st.text_input('Shimmer:DDA')
                nhr = st.text_input('NHR')
                hnr = st.text_input('HNR')
                rpde = st.text_input('RPDE')
                dfa = st.text_input('DFA')
                spread1 = st.text_input('spread1')
                spread2 = st.text_input('spread2')
                d2 = st.text_input('D2')
                ppe = st.text_input('PPE')

                # Code for Prediction
                if st.form_submit_button('Test Disease'):
                    try:
                        input_data = [
                            float(fo),
                            float(fhi),
                            float(flo),
                            float(jitter_percent),
                            float(jitter_abs),
                            float(rap),
                            float(ppq),
                            float(ddp),
                            float(shimmer),
                            float(shimmer_db),
                            float(apq3),
                            float(apq5),
                            float(apq),
                            float(dda),
                            float(nhr),
                            float(hnr),
                            float(rpde),
                            float(dfa),
                            float(spread1),
                            float(spread2),
                            float(d2),
                            float(ppe)
                        ]

                        diagnosis = parkinsons_prediction(input_data)

                        # Insert user data into the database
                        insert_user_data_parkinsons(user_id, [
                            float(fo),
                            float(fhi),
                            float(flo),
                            float(jitter_percent),
                            float(jitter_abs),
                            float(rap),
                            float(ppq),
                            float(ddp),
                            float(shimmer),
                            float(shimmer_db),
                            float(apq3),
                            float(apq5),
                            float(apq),
                            float(dda),
                            float(nhr),
                            float(hnr),
                            float(rpde),
                            float(dfa),
                            float(spread1),
                            float(spread2),
                            float(d2),
                            float(ppe),
                            diagnosis
                        ])
                        st.success('Diagnosis: ' + diagnosis)
                    except ValueError:
                        st.error('Please enter valid input values.')

        if selected == 'Disease Prediction':
            # Disease prediction code
            st.write('Disease Prediction')

            # Getting the input data from the user
            with st.form('disease_form'):
                # Allow user to select only 5 symptoms
                symptom1 = st.selectbox("Symptom 1", options=symptom_options)
                symptom2 = st.selectbox("Symptom 2", options=symptom_options)
                symptom3 = st.selectbox("Symptom 3", options=symptom_options)
                symptom4 = st.selectbox("Symptom 4", options=symptom_options)
                symptom5 = st.selectbox("Symptom 5", options=symptom_options)

            # Map predictions back to disease labels
                if st.form_submit_button('Predict Disease'):
                    input_data = [symptom1, symptom2,
                                symptom3, symptom4, symptom5]
                    input_vector = np.zeros(len(symptom_options))
                    for symptom in input_data:
                        if symptom in symptom_options:
                            input_vector[symptom_options.index(symptom)] = 1

                    input_vector = input_vector.reshape(1, -1)

                    try:
                        # Predict using all models
                        diagnosis_dt, diagnosis_rf, diagnosis_nb = disease_prediction_symp(
                            input_vector)

                        # Insert the predicted diseases into the database
                        # Insert the predicted diseases into the database
                        insert_disease_data(user_id, ', '.join(
                            input_data), diagnosis_dt, "Decision Tree")
                        insert_disease_data(user_id, ', '.join(
                            input_data), diagnosis_rf, "Random Forest")
                        insert_disease_data(user_id, ', '.join(
                            input_data), diagnosis_nb, "Naive Bayes")

                        # Display predictions
                        st.subheader('Predicted Disease using:')
                        st.write('Decision Tree: ' + diagnosis_dt)
                        st.write('Random Forest: ' + diagnosis_rf)
                        st.write('Naive Bayes: ' + diagnosis_nb)
                    except Exception as e:
                        st.error(f'Error: {str(e)}')


    # User Authentication
    else:

        # Initialize session state
        if 'user_data' not in st.session_state:
            st.session_state['user_data'] = {
            'logged_in': False, 'username': None, 'user_id': None}
        # Sidebar
        with st.sidebar:
            if not st.session_state['user_data']['logged_in']:
                login_tab, signup_tab = st.tabs(['Login', 'Signup'])

            with signup_tab:
                # Create a new account
                st.subheader('Signup')
                with st.form('signup_form'):
                    username = st.text_input(
                        'Username', placeholder='Enter your username')
                    password = st.text_input(
                        'Password', type='password', placeholder='Enter your password')
                    confirm_password = st.text_input(
                        'Confirm Password', type='password', placeholder='Confirm your password')

                    submit_button = st.form_submit_button('Signup')
                    if submit_button:
                        if not username or not password or not confirm_password:
                            st.warning('Please fill in all the fields.')
                        elif not re.match(r'^[a-zA-Z0-9_]{4,20}$', username):
                            st.warning(
                                'Username must be between 4 and 20 characters long and can only contain letters, numbers, and underscores.')
                        elif not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,20}$', password):
                            st.warning(
                                'Password must be between 8 and 20 characters long, contain at least one uppercase letter, one lowercase letter, one number, and one special character.')
                        elif password != confirm_password:
                            st.warning('Passwords do not match!')
                        elif user_exists(username):
                            st.warning('Username already exists!')
                        else:
                            register_user(username, password)
                            st.success('Account created successfully!.')
                            # Switch to login form
                            st.session_state.form_view = 'Login'
                            st.experimental_rerun()

            with login_tab:
                # Login to an existing account
                st.subheader('Login')
                with st.form('login_form'):
                    username = st.text_input(
                        'Username', placeholder='Enter your username')
                    password = st.text_input(
                        'Password', type='password', placeholder='Enter your password')

                    submit_button = st.form_submit_button('Login')
                    if submit_button:
                        if not username or not password:
                            st.warning('Please fill in all the fields.')
                        if verify_user(username, password):
                            st.success('Logged in successfully!')
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            user_id = connect_db().execute(
                                'SELECT id FROM users WHERE username = ?', (username,)).fetchone()[0]
                            st.session_state['user_id'] = user_id
                        else:
                            st.warning('Invalid username or password!')


# Main content
if 'logged_in' in st.session_state and st.session_state['logged_in']:
    # st.title(f'Welcome, {st.session_state["username"]}!')
    st.image('https://img.freepik.com/premium-photo/adorable-cartoon-character-wearing-doctors-coat-green-background-with-cute-expression_996993-76124.jpg?w=1380')

else:
    # Title
    st.title("🩺 Welcome to the Disease Prediction App 🩺")

    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; margin: 2rem 0;">
            <img src="https://img.freepik.com/free-photo/3d-cartoon-hospital-healthcare-scene_23-2151644103.jpg?t=st=1722880960~exp=1722884560~hmac=a61a6d9ed372738d4c8ce2e2f6f4657ccb74a10c7a838625d7bcb21351e9c814&w=900" style="width: 45%; margin-botton: 4rem;">
            <img src="https://img.freepik.com/free-photo/3d-cartoon-hospital-healthcare-scene_23-2151644107.jpg?t=st=1722881206~exp=1722884806~hmac=72f50559ca978cd028318ec483b42f4a04a4efb9a452fa2bc21cdfcd7d6d472d&w=900" style="width: 45%; margin-botton: 4rem;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # About section
    st.header("About Us")
    st.write("Our web application is designed to predict the likelihood of various diseases based on user input data. By leveraging advanced machine learning algorithms and a comprehensive database of medical information, we aim to provide accurate and reliable predictions to assist healthcare professionals and individuals in making informed decisions. **This app is not a substitute for professional medical advice.**")
    # gap
    st.write(" ")
    # Features section
    st.header("✨ Key Features ✨")
    st.write("- 🧠  Accurate disease prediction based on user input data")
    st.write("- 💻 User-friendly interface for easy data input")
    st.write("- 📚 Comprehensive database of medical information")
    st.write("- 🔄 Regularly updated with the latest medical research")
    st.write(" ")
    # Diseases Information Section
    st.header("🔍 Understanding the Diseases 🔍")

    # Diabetes
    st.subheader("🍬 Diabetes")
    st.write("Diabetes is a chronic condition that affects how your body regulates blood sugar (glucose). It occurs when your pancreas doesn't produce enough insulin, or when your body doesn't use insulin properly.")
    st.write("**Common symptoms:**")
    st.write("- Frequent urination 🚽")
    st.write("- Excessive thirst 💧")
    st.write("- Increased hunger 🍔")
    st.write("- Unexplained weight loss 📉")
    st.write("- Fatigue 😴")
    st.write("- Blurred vision 👀")
    st.write("- Slow-healing sores 🤕")
    st.write(" ")

    # Heart Attack
    st.subheader("❤️ Heart Attack")
    st.write("A heart attack occurs when blood flow to the heart is blocked, usually by a buildup of fat, cholesterol, and other substances in the arteries that supply blood to the heart (coronary arteries).")
    st.write("**Common symptoms:**")
    st.write(
        "- Chest pain or discomfort that may feel like pressure, squeezing, fullness, or pain 💔")
    st.write(
        "- Pain or discomfort that radiates to the left arm, jaw, back, neck, or stomach 😖")
    st.write("- Shortness of breath 🫁")
    st.write("- Cold sweats 😓")
    st.write("- Nausea 🤢")
    st.write("- Lightheadedness or dizziness 😵")
    st.write(" ")

    # Parkinson's Disease
    st.subheader("🚶‍♂️ Parkinson's Disease")
    st.write("Parkinson's disease is a brain disorder that leads to shaking, stiffness, and difficulty with walking, balance, and coordination. It happens when nerve cells (neurons) in a part of the brain that controls movement become impaired or die.")
    st.write("**Common symptoms:**")
    st.write("- Tremors (shaking) in the hands, arms, legs, jaw, or head 👋")
    st.write("- Slowed movement (bradykinesia) 🐢")
    st.write("- Rigid muscles, making it difficult to move and causing pain 뻣")
    st.write("- Impaired posture and balance 🤸‍♂️")
    st.write("- Loss of automatic movements, such as blinking or smiling 🙂")
    st.write("- Speech changes, such as speaking softly, quickly, slurring words, or hesitating before talking 🗣️")
    st.write("- Writing changes ✍️")
    st.write(" ")

    # Contact section
    st.header("📞 Contact Us 📞")
    st.write(
        "For any queries, suggestions, or feedback, please feel free to reach out to us:")
    with st.container():
        st.markdown(
            """
        <div class="contact-info">
            <h2>Contact Information</h2>
            <p><strong>Email:</strong> contact@example.com</p>
            <p><strong>Phone:</strong> +1 (555) 123-4567</p>
        </div>
        """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
    <style>
    h1 {
        color: #0072C6;
        font-weight: bold;
        text-align: center;
    }
    h2 {
        color: #0072C6;
        font-weight: bold;
    }
    p {
        font-size: 16px;
        line-height: 1.5;
    }
    .contact-info {
        border: 2px solid grey;
        padding: 20px;
        border-radius: 10px;
        color: #0072C6;
    }
    </style>
    """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
