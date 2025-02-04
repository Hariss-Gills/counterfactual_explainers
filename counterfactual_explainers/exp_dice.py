from importlib.resources import files
from tomllib import load

import pandas as pd
from dice_ml import Data, Dice, Model
from joblib import load as load_scikit_model
from keras.models import load_model as load_keras_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from counterfactual_explainers.data.preprocess_data import (
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)
from counterfactual_explainers.train_models import replace_empty_with_none


# TODO: Maybe I do not need to encode the features?
def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = replace_empty_with_none(config)
    results = []

    for dataset in config["dataset"]:
        if dataset == "compas":
            data = read_compas_dataset()
        else:
            data = read_dataset(dataset)

        for model_name in config["model"]:
            continuous_features = data["continuous_features"]
            categorical_features = data["categorical_features"]
            features = data["features"]
            target = data["target"]

            params_model = config["model"][model_name]
            params_dataset = config["dataset"][dataset]

            preprocessor, target_encoder = create_data_transformer(
                continuous_features,
                categorical_features,
            )

            encoded_features = preprocessor.fit_transform(features)
            # encoded_target = target_encoder.fit_transform(target)

            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=params_dataset["test_size"],
                random_state=params_model["random_state"][0],
                stratify=target,
            )
            combined_train_df = pd.concat([X_train, y_train], axis=1)
            dice_data_object = Data(
                dataframe=combined_train_df,
                continuous_features=continuous_features.tolist(),
                outcome_name=params_dataset["target_name"],
            )

            X_train = preprocessor.fit_transform(X_train)
            # y_train = target_encoder.fit_transform(y_train)

            # X_train_enc, X_test_enc, y_train_enc, y_test_enc = (
            #     train_test_split(
            #         encoded_features,
            #         encoded_target,
            #         test_size=params_dataset["test_size"],
            #         random_state=params_model["random_state"][0],
            #         stratify=encoded_target,
            #     )
            # )
            # NOTE: For an unexplained reason, the input df expects the
            # target column to be included. Documentation does not mention
            # this.
            # http://interpret.ml/DiCE/notebooks/DiCE_getting_started.html
            # combined_train_df = pd.concat([X_train, y_train], axis=1)

            if model_name == "RF":
                # backend = "sklearn"
                # model = load_scikit_model(
                #     f"counterfactual_explainers/models/{model_name}_{dataset}.pkl"
                # )
                # method = "random"
                model = RandomForestClassifier()
                model = model.fit(X_train, y_train)
                m = Model(model=model, backend="sklearn")
                exp = Dice(dice_data_object, m, method="random")
                print(X_test[1:2])
                e1 = exp.generate_counterfactuals(
                    X_test[1:2], total_CFs=2, desired_class="opposite"
                )
                e1.visualize_as_list(show_only_changes=False)

            elif model_name == "DNN":
                # Use NN methods for DICE
                backend = "TF2"
                model = load_keras_model(
                    f"counterfactual_explainers/models/{model_name}_{dataset}.keras"
                )
                method = "gradient"
            else:
                raise ValueError(
                    f"Model configuration for '{model_name}' not found."
                )
            # Generate CF's print them for now
            print(model)
            predictions = model.predict(X_test_enc)
            dice_model_object = Model(model=model, backend=backend)
            counterfactual_explainer = Dice(
                dice_data_object, dice_model_object, method=method
            )

            # Select a query instance
            query_instance = X_test_enc[0:1]

            # X_test_enc_df = pd.DataFrame(
            #     query_instance, columns=features.columns
            # )
            # print(X_test_enc_df)
            explanation = counterfactual_explainer.generate_counterfactuals(
                query_instance, total_CFs=4, desired_class="opposite"
            )

            # Visualize the counterfactuals
            explanation.visualize_as_dataframe(show_only_changes=True)

            # Measure metrics

    # csv_path = package / "results" / "training.csv"
    # df_results = pd.DataFrame(results)
    # df_results.to_csv(csv_path, index=False)
    # print(df_results)


if __name__ == "__main__":
    main()
