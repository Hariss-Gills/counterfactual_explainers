from importlib.resources import files
from tomllib import load

import pandas as pd
from dice_ml import Data
from joblib import load as load_scikit_model
from keras.models import load_model as load_keras_model
from sklearn.model_selection import train_test_split

from counterfactual_explainers.data.preprocess_data import (
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)


def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = replace_empty_with_none(config)
    results = []

    for dataset in config["dataset"]:
        # Load datasets
        if dataset == "compas":
            data = read_compas_dataset()
        else:
            data = read_dataset(dataset)

        for model_name in config["model"]:
            # Load models
            continuous_features = data["continuous_features"]
            categorical_features = data["categorical_features"]
            features = data["features"]
            target = data["target"]
            preprocessor, target_encoder = create_data_transformer(
                continuous_features,
                categorical_features,
            )
            params = config["model"][model_name]

            encoded_features = preprocessor.fit_transform(features)
            encoded_target = target_encoder.fit_transform(target)
            X_train, X_test, y_train, y_test = train_test_split(
                encoded_features,
                encoded_target,
                test_size=,
                random_state=params["random_state"][0],
            )
            dice_data_object = Data(dataframe=X_train, continuous_features=continuous_features, outcome_name=target)

            if model_name == "RF":
                # Use on NN method for DICE
                dice_model_object = Model(model=model, backend="sklearn")

            elif model_name == "DNN":
                # Use NN methods for DICE
                dice_model_object = Model(model=model, backend="sklearn")

            else:
                raise ValueError(
                    f"Model configuration for '{model_name}' not found."
                )
            # Generate CF's print them for now

            # Measure metrics

    csv_path = package / "results" / "training.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)
    print(df_results)


if __name__ == "__main__":
    main()
