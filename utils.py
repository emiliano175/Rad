import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def train_model():
    np.random.seed(42)
    n = 500
    age = np.random.randint(25, 85, n)
    gender = np.random.choice([0, 1], n)
    site = np.random.choice(['Breast', 'Head & Neck', 'Pelvis', 'Lung', 'Prostate'], n)
    diabetes = np.random.choice([0, 1], n, p=[0.8, 0.2])
    hypertension = np.random.choice([0, 1], n, p=[0.7, 0.3])
    asthma = np.random.choice([0, 1], n, p=[0.9, 0.1])

    site_encoded = pd.get_dummies(site, prefix='site')
    X = pd.DataFrame({'age': age, 'gender': gender, 'diabetes': diabetes,
                      'hypertension': hypertension, 'asthma': asthma})
    X = pd.concat([X, site_encoded], axis=1)

    def simulate_y(row):
        return pd.Series([
            int(row['age'] > 60 or row.get('site_Pelvis', 0)),
            int(row.get('site_Breast', 0) or row.get('site_Head & Neck', 0)),
            int(row.get('site_Pelvis', 0) or row['asthma'])
        ])

    y = X.apply(simulate_y, axis=1)
    y.columns = ['Fatigue', 'Skin Irritation', 'Nausea']

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
    model.fit(X, y)

    return model
