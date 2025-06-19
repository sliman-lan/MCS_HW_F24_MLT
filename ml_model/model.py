# Begin Of The Model Building
## Loan Prediction Model

### First importing needed libraries
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ------------------------ تحميل البيانات ------------------------
df = pd.read_csv('loan_prediction.csv')
df.columns = df.columns.str.strip()  # تنظيف أسماء الأعمدة

# ------------------------ تقسيم البيانات ------------------------
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# ------------------------data splitting ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)


# ------------------------ معالجة القيم المفقودة ------------------------
# الأعمدة الرقمية
num_imputer = SimpleImputer(strategy='mean')
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])

X_test[num_cols] = num_imputer.fit(X_test[num_cols])


# الأعمدة الفئوية
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = ['Gender', 'Married','Dependents', 'Self_Employed']
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

X_test[cat_cols] = cat_imputer.fit(X_test[cat_cols])

# ------------------------ المعالجة الأولية ------------------------
# تحويل الأنواع الفئوية
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status']
for col in cat_cols:
    X_train[col] = le.fit_transform(X_train[col])


ohe= OneHotEncoder(drop='first', sparse_output=False)
property_ohe = ohe.fit_transform(df[['Property_Area']])
df_ohe = pd.DataFrame(property_ohe, columns=ohe.get_feature_names_out(["Property_Area"]))

df = pd.concat([df.drop("Property_Area", axis=1), df_ohe], axis=1)



# ------------------------ هندسة الميزات ------------------------
# إضافة ميزات جديدة
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1e-6)  # تجنب القسمة على صفر


# معالجة القيم المتطرفة
X_train_transformed = np.log1p(X_train)  # Log Transformation
X_test_transformed = np.log1p(X_test)  # Log Transformation


smote = SMOTE(random_state=42, sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X_train_transformed, y_train)



model = RandomForestClassifier(
    n_estimators=100,  # عدد الأشجار
    max_depth=10,       # الحد الأقصى لعمق الشجرة
    random_state=42,
    # enable for best recal
    class_weight='balanced'  # لمعالجة عدم التوازن في الفئات
    
)
model.fit(X_res, y_res)



# ----------- التنبؤ والتقييم -----------
y_pred = model.predict(X_test)

print("----- model evaluation -----")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n Confusion Matrix :")
print(confusion_matrix(y_test, y_pred))
print("\n Classification Report :")
print(classification_report(y_test, y_pred))



# ------------------------ مصفوفة الارتباك ------------------------
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix')
plt.show()


# Cross Validation
scores = cross_val_score(model, X_res, y_res, cv=5, scoring="f1")
print(f"F1 Scores: {scores}")
print(f"Mean F1: {scores.mean()}")
print(f"STD F1: {scores.std()}")