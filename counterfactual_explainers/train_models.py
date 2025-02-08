import random
from importlib.resources import files
from pathlib import Path
from tomllib import load

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
    read_compas_dataset,
    read_dataset,
)


# NOTE: The literature review never tunes the model which is strange
# Also it only uses RandomizedSearchCV
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


def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = clean_config(config)
    results = []

    for dataset in config["dataset"]:
        if dataset == "compas":
            data = read_compas_dataset()
        else:
            data = read_dataset(dataset)

        for model_name in config["model"]:

            params_model = config["model"][model_name]
            params_dataset = config["dataset"][dataset]

            seed = params_model["classifier__random_state"][0]
            # random.seed(seed)
            # np.random.seed(seed)
            # tf.random.set_seed(seed)

            continuous_features = data["continuous_features"]
            categorical_features = data["categorical_features"]
            features = data["features"]
            target = data["target"]
            preprocessor, target_encoder = create_data_transformer(
                continuous_features,
                categorical_features,
            )

            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=params_dataset["test_size"],
                random_state=seed,
                stratify=target,
            )
            if model_name == "RF":
                model = RandomForestClassifier()
            elif model_name == "DNN":
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
            else:
                raise ValueError(
                    f"Model configuration for '{model_name}' not found."
                )

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
                # random_state=seed,
            )
            hyperparam_tuner.fit(
                X_train,
                y_train,
            )
            best_pipeline = hyperparam_tuner.best_estimator_
            y_pred_train = best_pipeline.predict(X_train)
            y_pred_test = best_pipeline.predict(X_test)
            result = {
                "dataset": dataset,
                "classifier": model_name,
                "accuracy_train": accuracy_score(y_train, y_pred_train),
                "accuracy_test": accuracy_score(y_test, y_pred_test),
                "f1_macro_train": f1_score(
                    y_train, y_pred_train, average="macro"
                ),
                "f1_macro_test": f1_score(
                    y_test, y_pred_test, average="macro"
                ),
                "f1_micro_train": f1_score(
                    y_train, y_pred_train, average="micro"
                ),
                "f1_micro_test": f1_score(
                    y_test, y_pred_test, average="micro"
                ),
            }
            results.append(result)

            # NOTE: Keras .save() is better for performance with KerasClassifier
            model_path = Path("./counterfactual_explainers/models")
            model_path.mkdir(parents=True, exist_ok=True)
            if model_name == "DNN":
                model_path = model_path / f"{model_name}_{dataset}.keras"
                best_model = best_pipeline.named_steps["classifier"]
                best_model.model_.save(model_path)
            else:
                model_path = model_path / f"{model_name}_{dataset}.pkl"
                dump(best_pipeline, model_path)

    results_path = Path("./counterfactual_explainers/results")
    results_path.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_path / "training.csv", index=False)
    print(df_results)


if __name__ == "__main__":
    main()
