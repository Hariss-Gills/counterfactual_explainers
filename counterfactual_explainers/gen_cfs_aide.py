"""A module for generating counterfactual explanations using the AIDE
framework.

This module provides functionality for:
- Loading pre-trained Keras models
- Preparing dataset-specific configurations for AIDE
- Generating counterfactual explanations using artificial immune networks
- Handling dataset encoding/decoding for explanations
- Saving counterfactual results and runtime metrics

Key components:
- generate_and_save_counterfactuals: Core AIDE explanation generation workflow
- generate_cfs_for_dataset: Dataset-specific explanation orchestration
- decode_df: Helper for decoding encoded feature representations
- DATASET_PARAMS: Pre-configured parameters for different datasets

Typical usage: TODO
"""

from time import perf_counter
from typing import Any

import pandas as pd
from keras.models import Model
from keras.models import load_model as load_keras_model
from sklearn.model_selection import train_test_split

from counterfactual_explainers.aide.aide_explain import init_var_optAINet
from counterfactual_explainers.aide.prepare_data import (
    decode_df,
    get_line_columns,
    get_prob_dict,
)
from counterfactual_explainers.aide.prepare_data import (
    read_adult_dataset as aide_read_adult,
)
from counterfactual_explainers.aide.prepare_data import (
    read_compas_dataset as aide_read_compas,
)
from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    get_output_path,
    read_config,
    read_dataset,
)

RESULTS_PATH = get_output_path()
MODELS_PATH = get_output_path("models")
DATASET_PARAMS = {
    "compas": {"affinity": 2.50, "population": 55},
    "adult": {"affinity": 3.00, "population": 50},
    "fico": {"affinity": 1.00, "population": 50},
    "german_credit": {"affinity": 0.1875, "population": 50},
}


def generate_and_save_counterfactuals(
    model: Model,
    X_test: pd.DataFrame,
    index_in_arr: int,
    aide_data_object: dict[str, Any],
    model_name: str,
    dataset_name: str,
    prob_dict: dict[str, float],
) -> None:
    """Generate and persist counterfactual explanations using AIDE.

    Args:
        model: Pretrained Keras model for prediction
        X_test: Test features DataFrame
        index_in_arr: Index of query instance in test set
        aide_data_object: Dataset-specific configuration dictionary
        model_name: Type of ML model being explained
        dataset_name: Name of dataset being used
        prob_dict: Probability dictionary for target classes

    """

    lime_coeffs_reorder = []
    df_out = pd.DataFrame(columns=get_line_columns(aide_data_object))
    db_file = RESULTS_PATH / "ignore_aide_demo.db"

    runtimes = []

    # NOTE: this num_required_cfs is very hard to control
    # but usually increasing the affinity_constant and pop_size
    # does the trick maybe I can try to scale affinity_constant since
    # it has more of an effect.

    # NOTE: I had to do merge the two classes for compas here.

    # NOTE: AIDE seems to be very stable. Hence it's almost better to generate 20
    # in one go.
    parameter_dict = {
        "sort_by": "distance",
        "use_mads": True,
        "problem_size": 1,
        "search_space": [0, 1],
        "max_gens": 5,
        "pop_size": DATASET_PARAMS[dataset_name]["population"],
        "num_clones": 10,
        "beta": 1,
        "num_rand": 2,
        "affinity_constant": DATASET_PARAMS[dataset_name]["affinity"],
        "stop_condition": 0.01,
        "new_cell_rate": 1.0,
    }

    start_time = perf_counter()
    result, df_out = init_var_optAINet(
        model,
        X_test,
        index_in_arr,
        aide_data_object,
        prob_dict,
        db_file,
        lime_coeffs_reorder,
        df_out,
        parameter_dict,
    )
    stop_time = perf_counter()
    runtime = stop_time - start_time

    decoded_cfs = decode_df(df_out, aide_data_object)

    if not df_out.empty:
        runtimes.append(
            {
                "Number of Required CFS": 20,
                "Runtime": runtime,
            }
        )
        # decoded_cfs.to_csv(
        #     RESULTS_PATH / f"cf_aide_{model_name}_{dataset_name}.csv",
        #     index=False,
        # )
        print(decoded_cfs)
        print(result)
    else:
        print("AIDE could not find Counterfactuals for query_instance")

    if runtimes:
        runtime_df = pd.DataFrame(runtimes).set_index("Number of Required CFS")
        runtime_csv_path = (
            RESULTS_PATH / f"runtime_aide_{model_name}_{dataset_name}.csv"
        )
        runtime_df.to_csv(runtime_csv_path)


def generate_cfs_for_dataset(
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    """Orchestrate counterfactual generation workflow for a dataset.

    Args:
        dataset_name: Name of dataset to process
        config: Configuration dictionary with parameters

    """

    data = read_dataset(config, dataset_name)
    params_dataset = config["dataset"][dataset_name]

    features = data["features"]
    target = data["target"]

    if dataset_name == "compas":
        aide_data_object = aide_read_compas()

    else:
        aide_data_object = aide_read_adult(dataset_name)

    # HACK: this is needed so the same query_instance
    # is chosen for dice and aide.
    model_name = "dnn"
    params_model = config["model"][model_name]
    seed = params_model["classifier__random_state"][0]

    _, X_test_df, _, _ = train_test_split(
        features,
        target,
        test_size=params_dataset["test_size"],
        random_state=seed,
        stratify=target,
    )

    _, X_test, _, y_test = train_test_split(
        aide_data_object["X"],
        aide_data_object["y"],
        test_size=params_dataset["test_size"],
        random_state=seed,
        stratify=aide_data_object["y"],
    )

    query_instance = X_test_df.sample(random_state=seed)
    index_in_arr = X_test_df.index.get_loc(query_instance.index[0])
    encoded_query_instance = X_test[index_in_arr]

    encoded_query_instance_df = pd.DataFrame(
        data=encoded_query_instance.reshape(1, -1),
        columns=aide_data_object["X_columns_with_dummies"],
    )

    decoded_query_instance_df = decode_df(
        encoded_query_instance_df, aide_data_object
    )

    y_query_instance = y_test.loc[query_instance.index[0]]
    query_instance_combined = decoded_query_instance_df.copy()
    query_instance_combined["target"] = y_query_instance

    # NOTE: This should be the same as dice query_instance
    query_instance_combined.to_csv(
        RESULTS_PATH
        / f"cf_aide_{model_name}_{dataset_name}_query_instance_combined.csv",
        index=False,
    )

    print("This is the query_instance")
    print(decoded_query_instance_df)

    model = load_keras_model(
        f"counterfactual_explainers/models/aide_{model_name}"
        f"_{dataset_name}.keras"
    )
    prob_dict = get_prob_dict(encoded_query_instance, model, aide_data_object)
    generate_and_save_counterfactuals(
        model,
        X_test,
        index_in_arr,
        aide_data_object,
        model_name,
        dataset_name,
        prob_dict,
    )


def main() -> None:
    """Main execution function for aide counterfactual generation."""
    config = clean_config(read_config())
    for dataset_name in config["dataset"]:
        generate_cfs_for_dataset(dataset_name, config)


if __name__ == "__main__":
    main()
