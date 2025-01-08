from importlib.resources import files
from tomllib import load

import pandas as pd
from joblib import dump
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from counterfactual_explainers.data.preprocess_data import (
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)


def replace_empty_with_none(data):
    if isinstance(data, dict):
        return {k: replace_empty_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_empty_with_none(item) for item in data]
    elif data == "":
        return None
    else:
        return data


# PARAMS = {
#     "RF": {
#         "n_estimators": [8, 16, 32, 64, 128, 256, 512, 1024],
#         "min_samples_split": [2, 0.002, 0.01, 0.05, 0.1, 0.2],
#         "min_samples_leaf": [1, 0.001, 0.01, 0.05, 0.1, 0.2],
#         "max_depth": [None, 2, 4, 6, 8, 10, 12, 16],
#         "class_weight": [None, "balanced"],
#         "random_state": [0],
#     },
#     # TODO: What are the best options to choose
#     "DNN": {
#         "model__dim_1": [1024, 512, 256, 128, 64, 32, 16, 8, 4],
#         "model__dim_2": [1024, 512, 256, 128, 64, 32, 16, 8, 4],
#         "model__activation_0": ["sigmoid", "tanh", "relu"],
#         "model__activation_1": ["sigmoid", "tanh", "relu"],
#         "model__activation_2": ["sigmoid", "tanh", "relu"],
#         "model__dropout_0": [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
#         "model__dropout_1": [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
#         "model__dropout_2": [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
#         # "batch_size": [32, 64, 128],
#         # "epochs": [50, 100],
#         # "optimizer": ["adam", "sgd", "rmsprop"],
#         # "optimizer__learning_rate": [0.001, 0.01],
#         "random_state": [0],
#     },
# }
#
# RANDOM_STATE = 0


# NOTE: The paper never tunes the model which is strange
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
            preprocessor, target_encoder = create_data_transformer(
                continuous_features,
                categorical_features,
            )
            params_model = config["model"][model_name]
            params_dataset = config["dataset"][dataset]

            encoded_features = preprocessor.fit_transform(features)
            encoded_target = target_encoder.fit_transform(target)
            X_train, X_test, y_train, y_test = train_test_split(
                encoded_features,
                encoded_target,
                test_size=params_dataset["test_size"],
                random_state=params_model["random_state"][0],
                stratify=encoded_target,
            )
            if model_name == "RF":
                model = RandomForestClassifier()
            elif model_name == "DNN":
                num_labels = len(target_encoder.classes_)
                model = KerasClassifier(
                    build_dnn,
                    loss=(
                        "binary_crossentropy"
                        if num_labels <= 2
                        else "categorical_crossentropy"
                    ),
                    optimizer="adam",
                    dim_0=X_train.shape[1],
                    dim_out=1 if num_labels <= 2 else num_labels,
                )
            else:
                raise ValueError(
                    f"Model configuration for '{model_name}' not found."
                )

            print(model_name)
            hyperparam_tuner = RandomizedSearchCV(
                estimator=model,
                param_distributions=params_model,
                n_iter=100,
                cv=5,
                scoring="f1_macro",
                n_jobs=-1,
            )
            hyperparam_tuner.fit(
                X_train,
                y_train,
            )
            best_model = hyperparam_tuner.best_estimator_
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
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
            model_path = csv_path = package / "models"
            if model_name == "DNN":
                model_path = model_path / f"{model_name}_{dataset}.keras"
                best_model.model_.save(model_path)
            else:
                model_path = model_path / f"{model_name}_{dataset}.pkl"
                dump(best_model, model_path)

    csv_path = package / "results" / "training.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)
    print(df_results)


if __name__ == "__main__":
    main()
