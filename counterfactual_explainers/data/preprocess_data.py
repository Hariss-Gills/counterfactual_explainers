import pandas as pd
import numpy as np
import toml
from importlib.resources import path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder


def read_dataset(dataset_name):
    with path('counterfactual_explainers.data', 'dataset_config.toml') as toml_path:
        config = toml.load(toml_path)

    if dataset_name not in config:
        raise ValueError(f"Dataset configuration for '{dataset_name}' not found.")

    dataset_config = config[dataset_name]
    drop_columns = dataset_config.get("drop_columns", [])
    na_values = dataset_config.get("na_values", None)
    target_name = dataset_config.get("target_name")
    
    with path('counterfactual_explainers.data.raw_data', f"{dataset_name}.csv") as csv_path:
        df = pd.read_csv(csv_path, skipinitialspace=True, na_values=na_values)

    df.drop(columns=drop_columns, errors="ignore", inplace=True)

    return df, target_name


# TODO: Should this dataset really be this manipulated?
def get_compas_dataset():
    class_name = 'class'
    with path('counterfactual_explainers.data.raw_data', "compas-scores-two-years.csv") as csv_path:
        df = pd.read_csv(csv_path, skipinitialspace=True)

    columns = ['age',
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
def create_data_transformer(continuous_features, categorical_features, scaler='minmax', encode='onehot'):
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
