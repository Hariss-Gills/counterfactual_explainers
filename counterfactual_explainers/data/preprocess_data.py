import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder

def get_adult_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?')
    df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
    return df, class_name

def get_german_dataset(filename):
    class_name = 'default'
    df = pd.read_csv(filename, skipinitialspace=True)
    return df, class_name

def get_fico_dataset(filename):
    class_name = 'RiskPerformance'
    df = pd.read_csv(filename, skipinitialspace=True)
    return df, class_name

def get_compas_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace=True)

    # TODO: Should this dataset really be this manipulated?
    columns = ['age',  # 'age_cat',
               'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
    df['class'] = df['score_text']
    df.drop(['c_jail_in', 'c_jail_out', 'decile_score', 'score_text'], axis=1, inplace=True)

    return df, class_name



#TODO: Maybe using KNNImputer and IterativeImputer for such rows would be better?
def preprocess_features_and_target(continuous_features, categorical_features, scaler='minmax', encode='onehot'):
    """
    Creates a Scikit-learn pipeline for preprocessing continuous and categorical features.

    Parameters:
        continuous_features (list): Indices or names of continuous features.
        categorical_features (list): Indices or names of categorical features.
        scaler (str): Type of scaler to use ('minmax' or 'standard').
        encode (str): Encoding method for categorical features ('onehot' or 'ordinal').

    Returns:
        Pipeline: A Scikit-learn pipeline for preprocessing.
    """
    
    steps_cont = [('imputer', SimpleImputer(strategy='mean'))]
    
    if scaler == 'minmax':
        steps_cont.append(('scaler', MinMaxScaler()))
    elif scaler == 'standard':
        steps_cont.append(('scaler', StandardScaler()))
    else:
        raise ValueError(f"Unknown scaler: {scaler}")
    
    continuous_transformer = Pipeline(steps=steps_cont)

    steps_cat = [('imputer', SimpleImputer(strategy='most_frequent'))]
    
    if encode == 'onehot':
            steps_cat.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
    elif encode == 'ordinal':
            steps_cat.append(('encoder', OrdinalEncoder()))
    else:
        raise ValueError(f"Unknown encoding: {encode}")
    
    categorical_transformer = Pipeline(steps=steps_cat)
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_features),
            ('categorical', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, LabelEncoder()



# df, name = get_adult_dataset("./raw_data/adult.csv")
# print(df.head())
# df, name = get_german_dataset("./raw_data/german_credit.csv")
# print(df.head())
# df, name = get_fico_dataset("./raw_data/fico.csv")
# print(df.head())
# df, name = get_compas_dataset("./raw_data/compas-scores-two-years.csv")
# print(df.head())
