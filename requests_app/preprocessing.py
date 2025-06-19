import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define your preprocessing steps
def preprocess_data(Applicant_Name, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    # Create a DataFrame for the input data
    data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    })


    ###################
    # Numerical Columns
    num_imputer = SimpleImputer(strategy='median')
    num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

    # Caterogical Columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = ['Gender', 'Married','Dependents', 'Self_Employed']
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

    # معالجة القيم المتطرفة
    data['ApplicantIncome'] = np.log1p(data['ApplicantIncome'])  # Log Transformation
    data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome']) 

   

    # تحويل الأنواع الفئوية
    le = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])


    # ohe= OneHotEncoder(drop='first', sparse_output=False)
    # property_ohe = ohe.fit_transform(data[['Property_Area']])
    # df_ohe = pd.DataFrame(property_ohe, columns=ohe.get_feature_names_out(["Property_Area"]))

    # data = pd.concat([data.drop("Property_Area", axis=1), df_ohe], axis=1)
    
    # Define encoder with categories explicitly to match training
    ohe = OneHotEncoder(categories=[["Rural", "Semiurban", "Urban"]], drop='first', sparse_output=False, handle_unknown='ignore')

    # Fit-transform on Property_Area
    property_ohe = ohe.fit_transform(data[['Property_Area']])

    # Ensure consistent column names
    expected_columns = ["Property_Area_Semiurban", "Property_Area_Urban"]
    df_ohe = pd.DataFrame(property_ohe, columns=expected_columns)

    # Merge with original data
    data = pd.concat([data.drop("Property_Area", axis=1), df_ohe], axis=1)
    ####################
   
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

    data['LoanToIncomeRatio'] = data['LoanAmount'] / data['TotalIncome']
    processed_data = data

    return processed_data
