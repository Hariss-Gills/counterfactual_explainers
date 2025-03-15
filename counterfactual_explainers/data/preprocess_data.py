"""A module for loading and preprocessing datasets based on configuration
files.

This module provides utilities to read dataset configurations from TOML
files, load and clean specified datasets, and create scikit-learn
preprocessing pipelines for machine learning tasks. It handles both categorical
and numerical features with configurable scaling/encoding strategies,
and preserves dataset metadata for downstream processing.

Key components:

    read_config: Loads and parses dataset configurations from TOML files

    read_dataset: Loads CSV data with configurable cleaning and feature
    splitting

    create_data_transformer: Builds customizable preprocessing pipelines

    DatasetDict: Typed dictionary structure for organized dataset components

Typical usage:

    Load configuration: config = read_config()

    Clean configuration: cleaned_config = clean_config(config)

    Load dataset: data = read_dataset(cleaned_config, 'compas')

    Create preprocessor: preprocessor, encoder = create_data_transformer(...)

    Transform: X_transformed = preprocessor.fit_transform(data['features'])

    Encode target: y_encoded = encoder.fit_transform(data['target'])
    """

from importlib.resources import files
from pathlib import Path
from tomllib import load
from typing import Any, Literal, TypedDict

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


class DatasetDict(TypedDict):
    """A typed dictionary containing organized dataset components and metadata.

    Attributes:
        target: Pandas Series containing the target variable
        features: Pandas DataFrame containing feature variables
        categorical_features: Index of categorical feature names
        continuous_features: Index of numerical feature names
        non_act_features: List of non-actionable feature names
        encode: Categorical encoding strategy from config (onehot/ordinal/None)
        scaler: Feature scaling strategy from config (minmax/standard/None)
    """

    target: pd.Series
    features: pd.DataFrame
    categorical_features: pd.Index
    continuous_features: pd.Index
    non_act_features: list[str]
    encode: Literal["onehot", "ordinal"] | None
    scaler: Literal["minmax", "standard"] | None


def read_config(config_name: str = "config.toml") -> dict[str, Any]:
    """Read a TOML configuration file from a package.

    Loads and parses a TOML configuration file from the specified package
    or determines the calling package automatically if not specified.

    Args:
        config_name: Name of the configuration file to read. Defaults to
        "config.toml".
        package: Package name where the config file is located. If None,
        will use the current package.

    Returns:
        A dictionary containing the parsed TOML configuration data.
        For example:
        {
            'dataset': {
                'compas': {'target_name': 'class', 'drop_columns': ['id']},
                'adult': {'target_name': 'income', 'na_values': '?'}
            },
            'model': {
                'random_state': 42,
                'test_size': 0.2
            }
        }
    """
    top_level_package = __package__.split(".")[0]
    package = files(top_level_package)
    config_path = package / config_name

    with config_path.open("rb") as file:
        return load(file)


def clean_config(data: dict[str, Any]) -> dict[str, Any]:
    """Clean configuration data by converting empty strings to None.

    Recursively processes dictionaries, lists, and other data structures,
    replacing empty strings with None values.

    Args:
        data: The configuration data to clean. Can be a dictionary, list,
              or any other data type.

    Returns:
        The cleaned configuration data with the same structure as the input,
        but with empty strings replaced by None.
        For example:
        clean_config({"name": "", "age": 30})
        {"name": None, "age": 30}
    """
    if isinstance(data, dict):
        return {k: clean_config(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_config(item) for item in data]
    elif data == "":
        return None
    else:
        return data


def get_dataframe(dataset_name: str, na_values: str) -> pd.DataFrame:
    """Load a dataset from package resources into a pandas DataFrame.

    Retrieves the specified dataset from either the 'cleaned_data' or
    'raw_data' directory within the package based on the dataset name.
    Handles path construction and CSV parsing with configurable NA
    value handling.

    Args:
        dataset_name: Name of the dataset to load (e.g., 'compas', 'adult').
        na_values: Additional strings to recognize as NA/NaN, passed to
        pandas.read_csv.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    package = files(__package__)
    if dataset_name == "compas":
        dir = "cleaned_data"
    else:
        dir = "raw_data"

    csv_path = package / dir / f"{dataset_name}.csv"
    with csv_path.open("rb") as file:
        return pd.read_csv(file, skipinitialspace=True, na_values=na_values)


# NOTE: I'm kinda worried that I may need to change this if I want
# to release this as a package.
def get_output_path(path_name: str = "results") -> Path:
    """Create and return a directory path for output files within the package.

    Ensures the directory exists by creating it if necessary, including
    any parent directories. Defaults to creating a 'results' directory in
    the top-level package.

    Args:
        path_name: Name of the output directory to create.
        Defaults to "results".

    Returns:
        Path: Path object pointing to the created output directory.
    """
    top_level_package = __package__.split(".")[0]
    package = Path(top_level_package)
    output_path = package / path_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def read_dataset(config: dict[str, Any], dataset_name: str) -> DatasetDict:
    """Read and preprocess a dataset based on configuration.

    Loads a dataset from a CSV file according to the provided configuration,
    performs initial preprocessing, and splits it into features and target.

    Args:
        config: Configuration dictionary containing dataset settings.
               Expected to have a 'dataset' key with dataset-specific
               configurations.
        dataset_name: Name of the dataset to load (e.g., 'compas', 'adult').

    Returns:
        A DatasetDict
    """
    dataset_config = config["dataset"][dataset_name]
    drop_columns = dataset_config.get("drop_columns", [])
    na_values = dataset_config.get("na_values", None)
    target_label = dataset_config.get("target_name")
    non_act_features = dataset_config.get("non_act_cols")
    df = get_dataframe(dataset_name, na_values)
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
        "encode": dataset_config.get("encode"),
        "scaler": dataset_config.get("scaler"),
    }


# NOTE: Maybe using KNNImputer and IterativeImputer
# for such rows would be better?
# WARNING: I need to return a ColumnTransformer here instead of a pipeline
# since the dataset stats are independant of a classifier which a
# pipeline needs to determine number of classes.
def create_data_transformer(
    continuous_features: list[str],
    categorical_features: list[str],
    scaler: Literal["minmax", "standard"] | None = "minmax",
    encode: Literal["onehot", "ordinal"] | None = "onehot",
) -> tuple[ColumnTransformer, LabelEncoder]:
    """Creates data transformation pipelines for machine learning
    preprocessing.

    This function builds separate preprocessing pipelines for continuous and
    categorical features, then combines them into a single ColumnTransformer.

    Args:
        continuous_features: List of column names for continuous features.
        categorical_features: List of column names for categorical features.
        scaler: Scaling method for continuous features. Options are "minmax"
            for MinMaxScaler, "standard" for StandardScaler, or None for
            no scaling.
            Defaults to "minmax".
        encode: Encoding method for categorical features. Options are "onehot"
            for OneHotEncoder, "ordinal" for OrdinalEncoder, or None for
            no encoding.
            Defaults to "onehot".

    Returns:
        A tuple containing:
            - preprocessor: A ColumnTransformer combining the continuous and
              categorical preprocessing pipelines.
            - label_encoder: A LabelEncoder instance for target variable
            encoding.
    """
    steps_cont = [("imputer", SimpleImputer(strategy="mean"))]

    if scaler is not None:
        if scaler == "minmax":
            steps_cont.append(("scaler", MinMaxScaler()))
        elif scaler == "standard":
            steps_cont.append(("scaler", StandardScaler()))

    continuous_transformer = Pipeline(steps=steps_cont)

    steps_cat = [("imputer", SimpleImputer(strategy="most_frequent"))]

    if encode is not None:
        if encode == "onehot":
            steps_cat.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            )
        elif encode == "ordinal":
            steps_cat.append(("encoder", OrdinalEncoder()))

    categorical_transformer = Pipeline(steps=steps_cat)
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, continuous_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, LabelEncoder()
