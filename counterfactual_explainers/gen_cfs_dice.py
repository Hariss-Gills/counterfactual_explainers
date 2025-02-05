from importlib.resources import files
from pathlib import Path
from tomllib import load

import pandas as pd
from dice_ml import Data, Dice, Model
from dice_ml.dice import UserConfigValidationException
from joblib import load as load_scikit_model
from keras.models import load_model as load_keras_model
from sklearn.model_selection import train_test_split

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    read_compas_dataset,
    read_dataset,
)


# NOTE: For now genrate using RF model only
# TODO: measure runtime
def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = clean_config(config)

    for dataset in config["dataset"]:
        if dataset == "compas":
            data = read_compas_dataset()
            desired_class = 2
        else:
            data = read_dataset(dataset)
            desired_class = "opposite"

        for model_name in config["model"]:
            params_model = config["model"][model_name]
            params_dataset = config["dataset"][dataset]

            seed = params_model["classifier__random_state"][0]

            continuous_features = data["continuous_features"]
            categorical_features = data["categorical_features"]
            non_act_features = data["non_act_features"]
            features = data["features"]
            target = data["target"]

            all_feat = features.columns.values.tolist()
            act_feat = list(set(all_feat) - set(non_act_features))
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=params_dataset["test_size"],
                random_state=seed,
                stratify=target,
            )
            if model_name == "RF":
                model = load_scikit_model(
                    f"counterfactual_explainers/models/{model_name}_{dataset}.pkl"
                )
                combined_train_df = pd.concat([X_train, y_train], axis=1)
                dice_data_object = Data(
                    dataframe=combined_train_df,
                    continuous_features=continuous_features.tolist(),
                    outcome_name=params_dataset["target_name"],
                )
                dice_model_object = Model(model=model, backend="sklearn")
                dice_exp = Dice(
                    dice_data_object, dice_model_object, method="genetic"
                )

                # WARNING: this instance for fico always fails but
                # should probably be kept. Also if num_required_cfs = 1
                # it throws a diffrent error hence catch that here too
                query_instance = X_test.sample(random_state=seed)

                for num_required_cfs in range(1, 21):
                    try:
                        explanation = dice_exp.generate_counterfactuals(
                            query_instance,
                            total_CFs=num_required_cfs,
                            desired_class=desired_class,
                            features_to_vary=act_feat,
                        )
                        cfs_for_all_queries = explanation.cf_examples_list
                        cfs = cfs_for_all_queries[
                            0
                        ]  # Always have one query instance
                        results_path = Path(
                            "./counterfactual_explainers/results"
                        )
                        results_path.mkdir(parents=True, exist_ok=True)
                        # df_results = pd.DataFrame(results)
                        cfs.final_cfs_df.to_csv(
                            results_path
                            / f"cf_dice_{model_name}_{dataset}_{num_required_cfs}.csv",
                            index=False,
                        )
                        # print(df_results)
                    except UserConfigValidationException as error:
                        if "No counterfactuals found" in str(error):
                            print(
                                f"DICE could not find Counterfactuals for {query_instance}"
                            )
                    except IndexError as error:
                        print(
                            f"fico failure when num_required_cfs={num_required_cfs}"
                        )

    #         # NOTE: For an unexplained reason, the input df expects the
    #         # target column to be included. Documentation does not mention
    #         # this.
    #         # http://interpret.ml/DiCE/notebooks/DiCE_getting_started.html
    #         # combined_train_df = pd.concat([X_train, y_train], axis=1)
    #
    #         if model_name == "RF":
    #             # backend = "sklearn"
    #             # model = load_scikit_model(
    #             #     f"counterfactual_explainers/models/{model_name}_{dataset}.pkl"
    #             # )
    #             # method = "random"
    #             model = RandomForestClassifier()
    #             model = model.fit(X_train, y_train)
    #             m = Model(model=model, backend="sklearn")
    #             exp = Dice(dice_data_object, m, method="random")
    #             print(X_test[1:2])
    #             e1 = exp.generate_counterfactuals(
    #                 X_test[1:2], total_CFs=2, desired_class="opposite"
    #             )
    #             e1.visualize_as_list(show_only_changes=False)
    #
    #         elif model_name == "DNN":
    #             # Use NN methods for DICE
    #             backend = "TF2"
    #             model = load_keras_model(
    #                 f"counterfactual_explainers/models/{model_name}_{dataset}.keras"
    #             )
    #             method = "gradient"
    #         else:
    #             raise ValueError(
    #                 f"Model configuration for '{model_name}' not found."
    #             )
    #         # Generate CF's print them for now
    #         print(model)
    #         predictions = model.predict(X_test_enc)
    #         dice_model_object = Model(model=model, backend=backend)
    #         counterfactual_explainer = Dice(
    #             dice_data_object, dice_model_object, method=method
    #         )
    #
    #         # Select a query instance
    #         query_instance = X_test_enc[0:1]
    #
    #         # X_test_enc_df = pd.DataFrame(
    #         #     query_instance, columns=features.columns
    #         # )
    #         # print(X_test_enc_df)
    #         explanation = counterfactual_explainer.generate_counterfactuals(
    #             query_instance, total_CFs=4, desired_class="opposite"
    #         )
    #
    #         # Visualize the counterfactuals
    #         explanation.visualize_as_dataframe(show_only_changes=True)
    #
    #         # Measure metrics
    #
    # # csv_path = package / "results" / "training.csv"
    # # df_results = pd.DataFrame(results)
    # # df_results.to_csv(csv_path, index=False)
    # # print(df_results)
    #
    #


if __name__ == "__main__":
    main()
