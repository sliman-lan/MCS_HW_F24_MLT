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