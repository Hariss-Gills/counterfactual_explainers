"""A module for generating dataset statistics and
preprocessing pipeline metrics.

This module provides functionality to analyze datasets and their preprocessing
pipelines, calculating key metrics about feature types, encoding results, and
label distributions. Results are exported to CSV for easy reporting.

Key functions:
    get_pipeline_stats: Calculates preprocessing pipeline metrics for a dataset
    main: Main execution flow that processes all configured datasets

Typical usage: TODO
"""

import pandas as pd

from counterfactual_explainers.data.preprocess_data import (
    DatasetDict,
    clean_config,
    create_data_transformer,
    get_output_path,
    read_config,
    read_dataset,
)

RESULTS_PATH = get_output_path()


def get_pipeline_stats(data: DatasetDict) -> dict[str, int | None]:
    """Analyzes a dataset and its preprocessing pipeline to
    calculate key metrics.

    Args:
        data: Dataset dictionary containing features, target, and metadata from
              read_dataset. Must include:
              - continuous_features: List of continuous feature names
              - categorical_features: List of categorical feature names
              - features: Full feature DataFrame
              - target: Target variable Series
              - non_act_features: List of non-actionable feature names
              - encode: Encoding strategy used for categorical features
              - scaler: Scaling strategy used for continuous features

    Returns:
        Dictionary containing calculated metrics with keys:
        - 'Number of Records': Total number of instances in the dataset
        - 'Number of Features': Original number of features before encoding
        - 'Number of Continuous Features': Count of numerical features
        - 'Number of Categorical Features': Count of categorical features
        - 'Number of Actionable Features': Features available for modification
        - 'Number of Encoded Features': Resulting features after encoding
        - 'Number of Labels': Distinct classes in the target variable
    """
    continuous_features = data["continuous_features"]
    categorical_features = data["categorical_features"]
    features = data["features"]
    target = data["target"]
    non_act_features = data["non_act_features"]
    encoder = data["encode"]
    scaler = data["scaler"]

    preprocessor, target_encoder = create_data_transformer(
        continuous_features,
        categorical_features,
        encoder,
        scaler,
    )

    num_of_records, num_of_features = features.shape
    num_of_cont = len(continuous_features)
    num_of_cat = len(categorical_features)
    num_of_act = num_of_features - len(non_act_features)
    num_encoded_features = None

    if encoder:
        encoded_features = preprocessor.fit_transform(features)
        num_encoded_features = encoded_features.shape[1]

    target_encoder.fit_transform(target)
    num_labels = len(target_encoder.classes_)

    return {
        "Number of Records": num_of_records,
        "Number of Features": num_of_features,
        "Number of Continuous Features": num_of_cont,
        "Number of Categorical Features": num_of_cat,
        "Number of Actionable Features": num_of_act,
        "Number of Encoded Features": num_encoded_features,
        "Number of Labels": num_labels,
    }


def main():
    """Main execution function that processes all datasets in configuration.

    Reads and cleans configuration, processes each dataset listed in the
    configuration file, calculates pipeline statistics, and exports results
    to a CSV file in the results directory.
    """
    config = clean_config(read_config())
    results = []
    for dataset_name in config["dataset"]:
        data = read_dataset(config, dataset_name)
        stats = get_pipeline_stats(data)
        stats["Dataset Name"] = dataset_name
        results.append(stats)

    df = pd.DataFrame(results)
    columns_order = ["Dataset Name"] + df.columns.drop("Dataset Name").tolist()
    df = df[columns_order]
    print(df)
    df.to_csv(RESULTS_PATH / "dataset_stats.csv", index=False)


if __name__ == "__main__":
    main()
