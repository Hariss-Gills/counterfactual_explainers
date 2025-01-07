from importlib.resources import files
from tomllib import load

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def read_dataset(dataset_name):
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    if dataset_name not in config["dataset"]:
        raise ValueError(
            f"Dataset configuration for '{dataset_name}' not found."
        )

    dataset_config = config["dataset"][dataset_name]
    drop_columns = dataset_config.get("drop_columns", [])
    na_values = dataset_config.get("na_values", None)
    target_label = dataset_config.get("target_name")
    non_act_features = dataset_config.get("non_act_cols")

    package = files("counterfactual_explainers.data.raw_data")
    csv_path = package / f"{dataset_name}.csv"
    with csv_path.open("rb") as file:
        df = pd.read_csv(file, skipinitialspace=True, na_values=na_values)

    df.drop(columns=drop_columns, errors="ignore", inplace=True)

    target = df[target_label]
    features = df.drop(columns=[target_label])
    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns
    continuous_features = features.select_dtypes(
        exclude=["object", "category"]
    ).columns
    return {
        "target": target,
        "features": features,
        "categorical_features": categorical_features,
        "continuous_features": continuous_features,
        "non_act_features": non_act_features,
    }


# TODO: Should this dataset really be this manipulated?
def read_compas_dataset():
    target_label = "class"
    package = files("counterfactual_explainers.data.raw_data")
    csv_path = package / "compas-scores-two-years.csv"
    with csv_path.open("rb") as file:
        df = pd.read_csv(file, skipinitialspace=True)

    columns = [
        "age",
        "sex",
        "race",
        "priors_count",
        "days_b_screening_arrest",
        "c_jail_in",
        "c_jail_out",
        "c_charge_degree",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
        "decile_score",
        "score_text",
    ]

    df = df[columns]
    df["days_b_screening_arrest"] = np.abs(df["days_b_screening_arrest"])
    df["c_jail_out"] = pd.to_datetime(df["c_jail_out"])
    df["c_jail_in"] = pd.to_datetime(df["c_jail_in"])
    df["length_of_stay"] = (df["c_jail_out"] - df["c_jail_in"]).dt.days
    df["length_of_stay"] = np.abs(df["length_of_stay"])
    df["length_of_stay"].fillna(
        df["length_of_stay"].value_counts().index[0], inplace=True
    )
    df["days_b_screening_arrest"].fillna(
        df["days_b_screening_arrest"].value_counts().index[0], inplace=True
    )
    df["length_of_stay"] = df["length_of_stay"].astype(int)
    df["days_b_screening_arrest"] = df["days_b_screening_arrest"].astype(int)
    df["class"] = df["score_text"]
    df.drop(
        ["c_jail_in", "c_jail_out", "decile_score", "score_text"],
        axis=1,
        inplace=True,
    )

    target = df[target_label]
    features = df.drop(columns=[target_label])
    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns
    continuous_features = features.select_dtypes(
        exclude=["object", "category"]
    ).columns

    non_act_features = ["age", "sex", "race"]

    return {
        "target": target,
        "features": features,
        "categorical_features": categorical_features,
        "continuous_features": continuous_features,
        "non_act_features": non_act_features,
    }


# TODO: Maybe using KNNImputer and IterativeImputer
# for such rows would be better?
def create_data_transformer(
    continuous_features, categorical_features, scaler="minmax", encode="onehot"
):
    """
    Creates a Scikit-learn pipeline for preprocessing continuous
    and categorical features.

    Parameters:
        continuous_features (list): Indices or names of continuous features.
        categorical_features (list): Indices or names of categorical features.
        scaler (str): Type of scaler to use ('minmax' or 'standard').
        encode (str): Encoding method for categorical features
        ('onehot' or 'ordinal').

    Returns:
        Pipeline: A Scikit-learn pipeline for preprocessing.
    """

    steps_cont = [("imputer", SimpleImputer(strategy="mean"))]

    if scaler == "minmax":
        steps_cont.append(("scaler", MinMaxScaler()))
    elif scaler == "standard":
        steps_cont.append(("scaler", StandardScaler()))
    else:
        raise ValueError(f"Unknown scaler: {scaler}")

    continuous_transformer = Pipeline(steps=steps_cont)

    steps_cat = [("imputer", SimpleImputer(strategy="most_frequent"))]

    if encode == "onehot":
        steps_cat.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
    elif encode == "ordinal":
        steps_cat.append(("encoder", OrdinalEncoder()))
    else:
        raise ValueError(f"Unknown encoding: {encode}")

    categorical_transformer = Pipeline(steps=steps_cat)
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, continuous_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, LabelEncoder()
