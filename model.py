import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def stroke_prediction(ag, hpt, htdis, mar, resi, glu, bmi, gen, work, smk):

    df = pd.read_csv("stroke_data.csv")

    # drop irrelevant id column
    df.drop("id", axis=1, inplace=True)

    # balance the dataset
    dataNoStrok = df[df.stroke != 1]
    dataYesStroke = df[df.stroke != 0]
    dataNoStrok = dataNoStrok.sample(300)
    df = pd.concat([dataNoStrok, dataYesStroke], axis=0)

    # function to detect outliers
    def detect_outliers(df, features):
        outlier_indices = []

        for c in features:
            # 1st quartile
            Q1 = np.percentile(df[c], 25)
            # 3rd quartile
            Q3 = np.percentile(df[c], 75)
            # IQR
            IQR = Q3 - Q1
            # Outlier step
            outlier_step = IQR * 1.5
            # detect outlier and their indeces
            outlier_list_col = df[(df[c] < Q1 - outlier_step)
                                  | (df[c] > Q3 + outlier_step)].index
            # store indeces
            outlier_indices.extend(outlier_list_col)

        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(
            i for i, v in outlier_indices.items() if v > 2)

        return multiple_outliers

    # drop outliers
    df = df.drop(detect_outliers(df, ['age', 'avg_glucose_level', 'bmi',
                 'hypertension', 'heart_disease', 'stroke']), axis=0).reset_index(drop=True)

    df['bmi'] = df['bmi'].fillna(0)

    # average value missing data imputation
    for i in range(0, 545):
        if (df['bmi'][i] == 0):
            if (df['gender'][i] == 'Male'):
                df['bmi'][i] = 28.594683544303823
            elif (df['gender'][i] == 'Female'):
                df['bmi'][i] = 29.035926055109936
            else:
                df['bmi'][i] = 28.854652338161664

    # Replacing string values from variables to numeric values
    ever_married_mapping = {'No': 0, 'Yes': 1}
    df['ever_married'] = df['ever_married'].map(ever_married_mapping)

    Residence_type_mapping = {'Rural': 0, 'Urban': 1}
    df['Residence_type'] = df['Residence_type'].map(Residence_type_mapping)

    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
    df['gender'] = df['gender'].map(gender_mapping)

    work_type_mapping = {'Self-employed': 0, 'Private': 1,
                         'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    df['work_type'] = df['work_type'].map(work_type_mapping)

    smoking_status_mapping = {'never smoked': 0,
                              'smokes': 1, 'formerly smoked': 2, 'Unknown': 3}
    df['smoking_status'] = df['smoking_status'].map(smoking_status_mapping)

    features = ['age',
                'hypertension',
                'heart_disease',
                'ever_married',
                'Residence_type',
                'avg_glucose_level',
                'bmi',
                'gender',
                'work_type',
                'smoking_status']

    label = ['stroke']

    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    logre = LogisticRegression()
    logre.fit(X_train, y_train)

    t_age = ag
    t_hypertension = hpt
    t_heart_disease = htdis
    t_ever_married = mar
    t_Residence_type = resi
    t_avg_glucose_level = glu
    t_bmi = bmi
    t_gender = gen
    t_work_type = work
    t_smoking_status = smk

    test_set = [[t_age, t_hypertension, t_heart_disease,
                 t_ever_married, t_Residence_type,
                 t_avg_glucose_level, t_bmi, t_gender, t_work_type,
                 t_smoking_status]]

    test = sc.transform(test_set)

    prediction = logre.predict(test)
    #prediction_prob = logre.predict_proba(test)

    #result_exist_probability = [prediction[0],round(prediction_prob[0][0]*100)]
    # result_exist_probability

    final_result = prediction[0]

    return final_result
