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
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)


# TODO: Metrics like runtime cannot be measured post-hoc
# So this will have to be done here
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

            model = None
            func = None
            backend = None
            method = None

            if model_name == "RF":
                model = load_scikit_model(
                    "counterfactual_explainers/models/"
                    f"{model_name}_{dataset}.pkl"
                )

                backend = "sklearn"
                method = "genetic"
                func = None

            if model_name == "DNN":
                preprocessor, target_encoder = create_data_transformer(
                    continuous_features=continuous_features,
                    categorical_features=categorical_features,
                )
                preprocessor.fit(features)
                print(preprocessor)
                cont_imputer = preprocessor.named_transformers_[
                    "continuous"
                ].named_steps["imputer"]
                cat_imputer = preprocessor.named_transformers_[
                    "categorical"
                ].named_steps["imputer"]

                new_feat_cont = cont_imputer.fit_transform(
                    features[continuous_features]
                )

                new_feat_cont = pd.DataFrame(
                    new_feat_cont,
                    columns=continuous_features,
                    index=features.index,
                )

                # HACK: this is dirty but
                # does the job
                if dataset != "fico":
                    new_feat_cat = cat_imputer.fit_transform(
                        features[categorical_features]
                    )

                    new_feat_cat = pd.DataFrame(
                        new_feat_cat,
                        columns=categorical_features,
                        index=features.index,
                    )
                    features = pd.concat([new_feat_cont, new_feat_cat], axis=1)
                else:
                    features = new_feat_cont
                transformed_target = target_encoder.fit_transform(target)
                target = pd.DataFrame(
                    transformed_target,
                    columns=[target.name],
                    index=target.index,
                )

                model = load_keras_model(
                    f"counterfactual_explainers/models/{model_name}"
                    f"_{dataset}.keras"
                )
                backend = "TF2"
                method = "gradient"
                func = "ohe-min-max"

            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
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
            results_path = Path("./counterfactual_explainers/results")
            results_path.mkdir(parents=True, exist_ok=True)
            query_instance.to_csv(
                results_path
                / f"cf_dice_{model_name}_{dataset}_query_instance.csv",
                index=False,
            )
            print(query_instance)
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
                    results_path = Path("./counterfactual_explainers/results")
                    results_path.mkdir(parents=True, exist_ok=True)
                    # df_results = pd.DataFrame(results)
                    print(cfs.final_cfs_df)

                    # NOTE: need this condtion since it doesn't
                    # do this check for deep learning methods.
                    if not cfs.final_cfs_df.empty:
                        cfs.final_cfs_df.to_csv(
                            results_path / f"cf_dice_{model_name}_{dataset}"
                            f"_{num_required_cfs}.csv",
                            index=False,
                        )
                    else:
                        print(
                            f"DICE could not find Counterfactuals"
                            f" for {query_instance}"
                        )
                except UserConfigValidationException as error:
                    if "No counterfactuals found" in str(error):
                        print(
                            "DICE could not find Counterfactuals"
                            f" for {query_instance}"
                        )
                except IndexError:
                    print(
                        "fico failure when num_required_cfs="
                        f"{num_required_cfs}"
                    )


if __name__ == "__main__":
    main()
