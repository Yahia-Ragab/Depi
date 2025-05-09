from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import math
from datetime import datetime

def process_liver_disease():

    # Label map for liver disease model
    label_map = {0: "C", 1: "CL", 2: "D"}

    # Feature names for liver disease model
    feature_names = [
        'N_Days', 'Drug', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
        'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
        'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage'
    ]

    # Load the trained model and scaler for liver disease model
    loaded_model = joblib.load('models/liver_disease/lgb_model.pkl')
    loaded_scaler = joblib.load('models/liver_disease/scaler.pkl')

    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Collect form data
            inputs = [
                request.form.get('n_days'),
                request.form.get('drug'),
                request.form.get('age'),
                request.form.get('sex'),
                request.form.get('ascites'),
                request.form.get('hepatomegaly'),
                request.form.get('spiders'),
                request.form.get('edema'),
                request.form.get('bilirubin'),
                request.form.get('cholesterol'),
                request.form.get('albumin'),
                request.form.get('copper'),
                request.form.get('alk_phos'),
                request.form.get('sgot'),
                request.form.get('tryglicerides'),
                request.form.get('platelets'),
                request.form.get('prothrombin'),
                request.form.get('stage')
            ]

            # Check for missing inputs
            if any(x is None or x == '' for x in inputs):
                error = "Please fill all fields."
                return render_template('liver_disease.html', prediction=prediction, error=error)

            # Convert inputs to appropriate types
            inputs = [float(x) if i not in [1, 3, 4, 5, 6, 7] else int(x) for i, x in enumerate(inputs)]

            # Convert age from days to years
            inputs[2] = inputs[2] / 365

            # Prepare the input features as a DataFrame
            features = pd.DataFrame([inputs], columns=feature_names)

            # Scale the features
            scaled_features = loaded_scaler.transform(features)

            # Predict using the model
            pred = loaded_model.predict(scaled_features)[0]
            prediction = f"Predicted Status: {label_map[pred]}"

        except ValueError as ve:
            error = f"Invalid input values. Please check your inputs. Details: {str(ve)}"
        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return render_template('liver_disease.html', prediction=prediction, error=error)


def process_length_of_stay():
    # Feature names for length of stay model
    feature_names = [
        'visit_date', 'readmission_count', 'gender', 'dialysis_or_end_stage_renal_disease', 
        'asthma', 'iron_deficiency', 'pneumonia', 'substance_dependence', 
        'major_psychological_disorder', 'depression', 'psychotherapy', 
        'pulmonary_fibrosis_or_other_lung_disease', 'malnutrition', 
        'hemoglobin_level', 'hematocrit', 'neutrophils', 'sodium', 
        'glucose', 'blood_urea_nitrogen', 'creatinine', 'bmi', 
        'pulse', 'respiration', 'secondary_diagnosis_non_icd9', 'facility_id'
    ]

    # Load the trained model and scaler for length of stay model
    loaded_model = joblib.load('models/length_of_stay/stacking_regressor.pkl')
    loaded_scaler = joblib.load('models/length_of_stay/scaler.pkl')

    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Use today's date as visit_date if not provided
            visit_date = datetime.now().strftime('%Y-%m-%d')
            
            # Collect form data
            inputs = {}
            for feature in feature_names:
                if feature == 'visit_date':
                    inputs[feature] = visit_date
                else:
                    value = request.form.get(feature)
                    if value is None or value == '':
                        error = f"Please fill the {feature} field."
                        return render_template('length_of_stay.html', prediction=prediction, error=error)
                    inputs[feature] = value

            # Create DataFrame from inputs
            features_df = pd.DataFrame([inputs])
            
            # Apply transformations for categorical variables
            
            # Gender transformation
            features_df['is_male'] = features_df['gender'].apply(lambda x: 1 if x == 'M' else 0).astype(int)
            features_df = features_df.drop(['gender'], axis=1)
            
            # Readmission count transformation
            features_df['rcount_5_or_more'] = (features_df['readmission_count'] == '5+').astype('int64')
            features_df['readmission_count'] = features_df['readmission_count'].apply(lambda x: 5 if x == '5+' else int(x)).astype(int)
            
            # Date handling
            def handle_date(date):
                date = pd.to_datetime(date)
                return date.year, date.month, date.day

            def infer_season(month):
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8]:
                    return 'summer'
                else:
                    return 'fall'

            date = features_df['visit_date'].apply(lambda x: pd.Series(handle_date(x)))
            date.columns = ['year', 'month', 'day']
            date['season'] = date['month'].apply(infer_season)
            date = pd.get_dummies(date, dtype=int)

            # Cyclical Encoding
            day_of_week = features_df["visit_date"].astype('datetime64[ns]').dt.dayofweek
            date["day_of_week_sin"] = day_of_week.apply(lambda x: math.sin(x * math.pi / 7))
            date["day_of_week_cos"] = day_of_week.apply(lambda x: math.cos(x * math.pi / 7))

            date["day_of_month_sin"] = date['day'].apply(lambda x: math.sin(x * math.pi / 31))
            date["day_of_month_cos"] = date['day'].apply(lambda x: math.cos(x * math.pi / 31))

            date["month_sin"] = date["month"].apply(lambda x: math.sin(x * math.pi / 12))
            date["month_cos"] = date["month"].apply(lambda x: math.cos(x * math.pi / 12))

            date = date.drop(['year', 'day', 'month'], axis=1)
            features_df = features_df.drop(['visit_date'], axis=1)
            features_df = pd.concat([features_df, date], axis=1)
            
            # Facility ID transformation
            facilities = pd.get_dummies(features_df[['facility_id']], dtype=int)
            features_df = pd.concat([features_df, facilities], axis=1)
            features_df = features_df.drop(['facility_id'], axis=1)
            
            # Convert remaining inputs to appropriate numeric types
            for col in features_df.columns:
                if col not in ['is_male', 'rcount_5_or_more'] and col.startswith(('facility_id_', 'season_')):
                    if col in ['hematocrit', 'neutrophils', 'sodium', 'glucose', 
                              'blood_urea_nitrogen', 'creatinine', 'bmi', 'respiration']:
                        features_df[col] = features_df[col].astype(float)
                    elif not col.startswith(('day_of_week_', 'day_of_month_', 'month_')):
                        features_df[col] = features_df[col].astype(int)

            # Ensure we have all the columns the scaler expects
            expected_columns = loaded_scaler.feature_names_in_
            for col in expected_columns:
                if col not in features_df.columns:
                    features_df[col] = 0  # Add missing columns with 0 values
            
            # Reorder columns to match what the scaler expects
            features_df = features_df[expected_columns]
            
            # Scale the features
            scaled_features = loaded_scaler.transform(features_df)

            # Predict using the model
            days_prediction = loaded_model.predict(scaled_features)[0]
            
            # Round to nearest whole day and convert to int for display
            days_prediction = round(days_prediction)
            
            prediction = f"Predicted Length of Stay: {days_prediction} days"

        except ValueError as ve:
            error = f"Invalid input values. Please check your inputs. Details: {str(ve)}"
        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return render_template('length_of_stay.html', prediction=prediction, error=error)

def process_diabetes():
    # Feature names for diabetes model
    feature_names = [
        'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
        'bmi', 'HbA1c_level', 'blood_glucose_level'
    ]

    # Load the trained model and scaler for diabetes model
    loaded_model = joblib.load('models/diabetes/voting_model.pkl')
    loaded_scaler = joblib.load('models/diabetes/scaler.pkl')

    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Collect form data
            inputs = {}
            for feature in feature_names:
                value = request.form.get(feature)
                if value is None or value == '':
                    error = f"Please fill the {feature} field."
                    return render_template('diabetes.html', prediction=prediction, error=error)
                inputs[feature] = value

            # Create DataFrame from inputs
            features_df = pd.DataFrame([inputs])
            
            # Convert numeric features to appropriate types
            features_df['age'] = features_df['age'].astype(float)
            features_df['hypertension'] = features_df['hypertension'].astype(int)
            features_df['heart_disease'] = features_df['heart_disease'].astype(int)
            features_df['bmi'] = features_df['bmi'].astype(float)
            features_df['HbA1c_level'] = features_df['HbA1c_level'].astype(float)
            features_df['blood_glucose_level'] = features_df['blood_glucose_level'].astype(int)
            
            # One-hot encode categorical variables
            # Gender encoding
            gender_dummies = pd.get_dummies(features_df['gender'], prefix='gender', dtype=int)
            
            # Smoking history encoding
            smoking_dummies = pd.get_dummies(features_df['smoking_history'], prefix='smoking', dtype=int)
            
            # Drop original categorical columns and concatenate with encoded ones
            features_df = features_df.drop(['gender', 'smoking_history'], axis=1)
            features_df = pd.concat([features_df, gender_dummies, smoking_dummies], axis=1)
            
            # Ensure all expected columns are present for the scaler
            expected_columns = loaded_scaler.feature_names_in_
            for col in expected_columns:
                if col not in features_df.columns:
                    features_df[col] = 0  # Add missing columns with default values
            
            # Reorder columns to match the scaler's expectations
            features_df = features_df[expected_columns]
            
            # Scale the features
            scaled_features = loaded_scaler.transform(features_df)

            # Predict using the model
            prediction_result = loaded_model.predict(scaled_features)[0]
            prediction = "Have Diabetes" if prediction_result == 1 else "Not Have Diabetes"

        except ValueError as ve:
            error = f"Invalid input values. Please check your inputs. Details: {str(ve)}"
        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return render_template('diabetes.html', prediction=prediction, error=error)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/liver_disease', methods=['GET', 'POST'])
def liver_disease():
    return process_liver_disease()

@app.route('/length_of_stay', methods=['GET', 'POST'])
def length_of_stay():
    today_date = datetime.now().strftime('%Y-%m-%d')
    if request.method == 'POST':
        return process_length_of_stay()
    return render_template('length_of_stay.html', prediction=None, error=None, today_date=today_date)

# Update the route handler
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    return process_diabetes()