"""A module for generating counterfactual explanations using the DICE
framework.

This module provides functionality for:
- Loading pre-trained machine learning models (Random Forest and DNN)
- Preprocessing data for model-specific requirements
- Generating counterfactual explanations using different DICE methods
- Saving counterfactual results and generation metrics
- Handling dataset-specific configuration and constraints

Key components:
- transform_data_for_dnn: Prepares data for deep neural network compatibility
- prepare_model_and_data: Loads appropriate model and preprocesses data
- generate_and_save_counterfactuals: Core DICE explanation generation workflow
- generate_cfs_for_dataset: Dataset-specific explanation orchestration

Typical usage: TODO
"""

from time import perf_counter
from typing import Any

import pandas as pd
from dice_ml import Data, Dice, Model
from dice_ml.dice import UserConfigValidationException
from joblib import load as load_scikit_model
from keras.models import load_model as load_keras_model
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
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


def transform_data_for_dnn(
    preprocessor: Pipeline,
    target_encoder: BaseEstimator,
    features: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    target: pd.Series,
    encoder: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Transform data for deep neural network models using specified
    preprocessing.

    Args:
        preprocessor: Fitted pipeline for feature preprocessing.
        target_encoder: Encoder for transforming target variable.
        features: Raw input features DataFrame.
        continuous_features: List of continuous feature names.
        categorical_features: List of categorical feature names.
        target: Raw target variable Series.
        encoder: Whether to apply categorical encoding.

    Returns:
        Tuple containing:
            - pd.DataFrame: Transformed features ready for DNN input
            - pd.Series: Transformed target variable
    """
    preprocessor.fit(features)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps[
        "imputer"
    ]
    cat_imputer = preprocessor.named_transformers_["categorical"].named_steps[
        "imputer"
    ]

    new_feat_cont = cont_imputer.fit_transform(features[continuous_features])
    new_feat_cont = pd.DataFrame(
        new_feat_cont,
        columns=continuous_features,
        index=features.index,
    )

    if encoder:
        new_feat_cat = cat_imputer.fit_transform(
            features[categorical_features]
        )
        new_feat_cat = pd.DataFrame(
            new_feat_cat,
            columns=categorical_features,
            index=features.index,
        )
        transformed_features = pd.concat([new_feat_cont, new_feat_cat], axis=1)
    else:
        transformed_features = new_feat_cont

    transformed_target_array = target_encoder.fit_transform(target)
    transformed_target = pd.DataFrame(
        transformed_target_array,
        columns=[target.name],
        index=target.index,
    )

    return transformed_features, transformed_target


def prepare_model_and_data(
    model_name: str,
    dataset_name: str,
    encoder: str | None,
    features: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    target: pd.Series,
) -> tuple[
    BaseEstimator | Any,
    str,
    str,
    str,
    pd.DataFrame,
    pd.Series,
]:
    """Prepare ML model and preprocess data based on model type.

    Args:
        model_name: Type of model ('rf' or 'dnn')
        dataset_name: Name of dataset being used
        encoder: Whether to apply categorical encoding
        features: Raw input features DataFrame
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
        target: Raw target variable Series

    Returns:
        Tuple containing:
            - Loaded ML model
            - DICE backend type (str or None)
            - DICE method (str or None)
            - DICE function type (str or None)
            - Processed features DataFrame
            - Processed target Series
    """
    model = None
    backend = None
    method = None
    func = None

    if model_name == "rf":
        model_path = MODELS_PATH / f"dice_{model_name}_{dataset_name}.pkl"
        model = load_scikit_model(model_path)
        backend = "sklearn"
        method = "genetic"

    elif model_name == "dnn":
        preprocessor, target_encoder = create_data_transformer(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
        )

        features, target = transform_data_for_dnn(
            preprocessor,
            target_encoder,
            features,
            continuous_features,
            categorical_features,
            target,
            encoder,
        )
        model_path = MODELS_PATH / f"dice_{model_name}_{dataset_name}.keras"
        model = load_keras_model(model_path)
        backend = "TF2"
        method = "gradient"
        func = "ohe-min-max"

    return model, backend, method, func, features, target


def generate_and_save_counterfactuals(
    dice_exp: Dice,
    query_instance: pd.DataFrame,
    act_features: list[str],
    desired_class: int | str,
    model_name: str,
    dataset_name: str,
    max_cfs: int = 20,
) -> None:
    """Generate and persist counterfactual explanations using DICE.

    Args:
        dice_exp: Initialized DICE explainer object
        query_instance: Input data point to explain
        act_features: List of features allowed to vary
        desired_class: Target class for counterfactuals
        model_name: Type of ML model being explained
        dataset_name: Name of dataset being used
        max_cfs: Maximum number of counterfactuals to generate

    Raises:
        UserConfigValidationException: If DICE configuration is invalid
        IndexError: If counterfactual generation fails for specific CF count
    """
    runtimes = []
    for num_required_cfs in range(1, max_cfs + 1):
        try:
            start_time = perf_counter()
            explanation = dice_exp.generate_counterfactuals(
                query_instance,
                total_CFs=num_required_cfs,
                desired_class=desired_class,
                features_to_vary=act_features,
            )
            stop_time = perf_counter()
            runtime = stop_time - start_time

            cfs_for_all_queries = explanation.cf_examples_list
            cfs = cfs_for_all_queries[0]

            if not cfs.final_cfs_df.empty:
                runtimes.append(
                    {
                        "Number of Required CFS": num_required_cfs,
                        "Runtime": runtime,
                    }
                )
                output_path = (
                    RESULTS_PATH
                    / f"cf_dice_{model_name}_{dataset_name}_{num_required_cfs}.csv"
                )
                # cfs.final_cfs_df.to_csv(output_path, index=False)
            else:
                print(
                    "DICE could not find counterfactuals"
                    " for query instance (empty result)"
                )

        except UserConfigValidationException as error:
            if "No counterfactuals found" in str(error):
                print(
                    "DICE could not find counterfactuals for"
                    " query instance (empty result)"
                )

        except IndexError as error:
            print(
                f"IndexError for {dataset_name} failure when num_required_cfs="
                f"{num_required_cfs}"
            )
    if runtimes:
        runtime_df = pd.DataFrame(runtimes).set_index("Number of Required CFS")
        runtime_csv_path = (
            RESULTS_PATH / f"runtime_dice_{model_name}_{dataset_name}.csv"
        )
        runtime_df.to_csv(runtime_csv_path)


def generate_cfs_for_dataset(
    dataset_name: str,
    config: dict[str, Any],
):
    """Orchestrate counterfactual generation workflow for a dataset.

    Args:
        dataset_name: Name of dataset to process
        config: Configuration dictionary with parameters
    """
    if dataset_name not in ["adult", "german_credit"]:
        data = read_dataset(config, dataset_name)
        params_dataset = config["dataset"][dataset_name]

        continuous_features = data["continuous_features"]
        categorical_features = data["categorical_features"]
        non_act_features = data["non_act_features"]

        features = data["features"]
        target = data["target"]
        encoder = data["encode"]

        if dataset_name == "compas":
            desired_class = 2
        else:
            desired_class = "opposite"

        for model_name in config["model"]:
            params_model = config["model"][model_name]
            seed = params_model["classifier__random_state"][0]
            all_feat = features.columns.values.tolist()
            act_feat = list(set(all_feat) - set(non_act_features))

            (
                model,
                backend,
                method,
                func,
                transformed_features,
                transformed_target,
            ) = prepare_model_and_data(
                model_name,
                dataset_name,
                encoder,
                features,
                continuous_features,
                categorical_features,
                target,
            )

            # NOTE: for sklearn models the data df doesn't have to be encoded
            X_train, X_test, y_train, y_test = train_test_split(
                transformed_features,
                transformed_target,
                test_size=params_dataset["test_size"],
                random_state=seed,
                stratify=target,
            )

            combined_train_df = pd.concat([X_train, y_train], axis=1)
            dice_data_object = Data(
                dataframe=combined_train_df,
                continuous_features=continuous_features.tolist(),
                outcome_name=params_dataset["target_name"],
            )
            dice_model_object = Model(model=model, backend=backend, func=func)
            dice_exp = Dice(dice_data_object, dice_model_object, method=method)

            # WARNING: this instance for fico RF mostly fails but
            # should probably be kept. Also if num_required_cfs = 1
            # it throws a different error, hence catch that here too.
            # compas DNN always fails too -> no CFs found.
            query_instance = X_test.sample(random_state=seed)
            sampled_index = query_instance.index[0]
            y_query_instance = y_test.loc[sampled_index]
            query_instance_combined = query_instance.copy()
            query_instance_combined["target"] = y_query_instance

            query_instance_combined.to_csv(
                RESULTS_PATH
                / f"cf_dice_{model_name}_{dataset_name}_query_instance_combined.csv",
                index=False,
            )

            print("This is the query_instance")
            print(query_instance_combined)

            generate_and_save_counterfactuals(
                dice_exp=dice_exp,
                query_instance=query_instance,
                act_features=act_feat,
                desired_class=desired_class,
                model_name=model_name,
                dataset_name=dataset_name,
            )


def main() -> None:
    """Main execution function for dice counterfactual generation."""
    config = clean_config(read_config())
    for dataset_name in config["dataset"]:
        generate_cfs_for_dataset(dataset_name, config)


if __name__ == "__main__":
    main()
