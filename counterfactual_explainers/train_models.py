"""A module for training and evaluating machine learning models with explainer integration.

This module provides end-to-end functionality for:
- Configuring and training Random Forest and DNN models
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation with multiple metrics
- Integration with different explainer frameworks (AIDE/DICE)
- Saving trained models and results

Key components:
- build_dnn: Constructs customizable neural network architectures
- train_model: Handles model training with hyperparameter tuning
- evaluate_model: Calculates performance metrics
- train_and_evaluate_for_dataset: Manages dataset-specific training workflows
- main: Coordinates end-to-end training process

Typical usage: TODO
"""

import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    create_data_transformer,
    get_output_path,
    read_config,
    read_dataset,
)

RESULTS_PATH = get_output_path()
MODELS_PATH = get_output_path("models")


# WARN: This has weird behaviour so do not
# use it
def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across multiple libraries.

    WARNING: May exhibit unexpected behavior with certain TensorFlow versions
    or when combined with GPU computations.

    Args:
        seed: Integer value to seed all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# NOTE: The literature review never tunes the model,
# and it only uses RandomizedSearchCV
def build_dnn(
    dim_0,
    dim_out,
    dim_1=128,
    dim_2=64,
    activation_0="relu",
    activation_1="relu",
    activation_2="relu",
    dropout_0=0.3,
    dropout_1=0.1,
    dropout_2=0.01,
):
    """Construct a deep neural network architecture with configurable layers.

    Creates a sequential neural network with dense layers and optional dropout.
    The final layer uses sigmoid activation for binary classification or
    softmax for multi-class (automatically determined during training).

    Args:
        dim_0: Input dimension size (must match preprocessed feature dimension)
        dim_out: Output dimension size (number of classes)
        dim_1: First hidden layer size, default 128
        dim_2: Second hidden layer size, default 64
        activation_0: Activation for input layer, default 'relu'
        activation_1: Activation for first hidden layer, default 'relu'
        activation_2: Activation for second hidden layer, default 'relu'
        dropout_0: Dropout rate after input layer, default 0.3
        dropout_1: Dropout rate after first hidden layer, default 0.1
        dropout_2: Dropout rate after second hidden layer, default 0.01

    Returns:
        Sequential: Uncompiled Keras Sequential model
    """
    model = Sequential()

    model.add(
        Dense(
            dim_0,
            activation=activation_0,
            kernel_initializer="uniform",
            input_dim=dim_0,
        )
    )
    if dropout_0 is not None:
        model.add(Dropout(dropout_0))

    model.add(
        Dense(dim_1, activation=activation_1, kernel_initializer="uniform")
    )
    if dropout_1 is not None:
        model.add(Dropout(dropout_1))

    model.add(
        Dense(dim_2, activation=activation_2, kernel_initializer="uniform")
    )
    if dropout_2 is not None:
        model.add(Dropout(dropout_2))

    model.add(Dense(dim_out, activation="sigmoid"))
    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    model: Any,
    params_model: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Train a machine learning model with hyperparameter tuning.

    Constructs a scikit-learn pipeline with preprocessing and classifier,
    then performs randomized search for hyperparameter optimization.

    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        preprocessor: Configured data preprocessing transformer
        model: Uninitialized classifier model
        params_model: Hyperparameter search space for RandomizedSearchCV
        seed: Random seed for reproducibility

    Returns:
        Pipeline: Best performing pipeline from hyperparameter search
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    hyperparam_tuner = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params_model,
        n_iter=100,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=seed,
        verbose=1,
    )

    hyperparam_tuner.fit(X_train, y_train)

    return hyperparam_tuner.best_estimator_


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate evaluation metrics for model performance.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - f1_macro: Macro-averaged F1 score
        - f1_micro: Micro-averaged F1 score
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
    }


def save_model(
    best_pipeline: Pipeline,
    model_name: str,
    dataset_name: str,
    explainer_type: str,
) -> Path:
    """Save trained model to disk in format appropriate for each model type.

    Args:
        best_pipeline: Trained scikit-learn pipeline
        model_name: Type of model ('RF' or 'DNN')
        dataset_name: Name of dataset used for training
        explainer_type: Type of explainer ('aide' or 'dice')

    Returns:
        Path: Location where model was saved
    """
    if model_name == "DNN":
        save_path = (
            MODELS_PATH / f"{model_name}_{explainer_type}_{dataset_name}.keras"
        )
        best_model = best_pipeline.named_steps["classifier"]
        best_model.model_.save(save_path)
    else:
        save_path = (
            MODELS_PATH / f"{model_name}_{explainer_type}_{dataset_name}.pkl"
        )
        dump(best_pipeline, save_path)

    return save_path


def train_and_evaluate_for_dataset(
    dataset_name: str, config: dict[str, Any], explainer_type: str
) -> list[dict[str, Any]]:
    """Orchestrate model training and evaluation for a single dataset.

    Args:
        dataset_name: Name of dataset to process
        config: Loaded configuration dictionary
        explainer_type: Type of explainer framework being used

    Returns:
        List of dictionaries containing training results for each model
    """
    data = read_dataset(config, dataset_name)
    params_dataset = config["dataset"][dataset_name]

    continuous_features = data["continuous_features"]
    categorical_features = data["categorical_features"]
    features = data["features"]
    target = data["target"]
    encoder = data["encode"]
    scaler = data["scaler"]

    results = []
    for model_name in config["model"]:
        params_model = config["model"][model_name]
        seed = params_model["classifier__random_state"][0]
        # set_random_seeds(seed)

        preprocessor, target_encoder = create_data_transformer(
            continuous_features,
            categorical_features,
            scaler,
            encoder,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=params_dataset["test_size"],
            random_state=seed,
            stratify=target,
        )

        if model_name == "rf":
            model = RandomForestClassifier()
        elif model_name == "dnn":
            encoded_features = preprocessor.fit_transform(features)
            target_encoder.fit_transform(target)
            num_labels = len(target_encoder.classes_)
            model = KerasClassifier(
                build_dnn,
                loss=(
                    "binary_crossentropy"
                    if num_labels <= 2
                    else "categorical_crossentropy"
                ),
                optimizer="adam",
                dim_0=encoded_features.shape[1],
                dim_out=1 if num_labels <= 2 else num_labels,
            )

        best_pipeline = train_model(
            X_train, y_train, preprocessor, model, params_model, seed
        )

        y_pred_train = best_pipeline.predict(X_train)
        y_pred_test = best_pipeline.predict(X_test)

        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)

        result = {
            "Dataset Name": dataset_name,
            "Classifier": model_name,
            "Accuracy Train": train_metrics["accuracy"],
            "Accuracy Test": test_metrics["accuracy"],
            "F1 Macro Train": train_metrics["f1_macro"],
            "F1 Macro Test": test_metrics["f1_macro"],
            "F1 Micro Train": train_metrics["f1_micro"],
            "F1 Micro Test": test_metrics["f1_micro"],
        }

        results.append(result)
        save_model(best_pipeline, model_name, dataset_name, explainer_type)

    return results


# NOTE: This has to be done since explainers
# use different tensorflow versions.
def parse_arguments():
    """Parse command line arguments for explainer framework selection.

    Returns:
        ArgumentParser: Configured argument parser with explainer type
    """
    parser = ArgumentParser(description="Train models with explainer type.")
    parser.add_argument(
        "--explainer",
        "-e",
        type=str,
        choices=["aide", "dice"],
        default="aide",
        help="Type of explainer to use (AIDE or DICE)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    explainer_type = args.explainer
    config = clean_config(read_config())
    results = []
    for dataset_name in config["dataset"]:
        dataset_results = train_and_evaluate_for_dataset(
            dataset_name, config, explainer_type
        )
        results.extend(dataset_results)

    df_results = pd.DataFrame(results)
    print(df_results)
    results_file = RESULTS_PATH / f"training_{explainer_type}.csv"
    df_results.to_csv(results_file, index=False)


if __name__ == "__main__":
    main()
