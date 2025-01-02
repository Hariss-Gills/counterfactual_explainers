from importlib.resources import path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from toml import load

from counterfactual_explainers.data.preprocess_data import (
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)

params = {
    "RF": {
        "n_estimators": [8, 16, 32, 64, 128, 256, 512, 1024],
        "min_samples_split": [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf": [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        "max_depth": [None, 2, 4, 6, 8, 10, 12, 16],
        "class_weight": [None, "balanced"],
        "random_state": [0],
    }
}


def main():
    with path(
        "counterfactual_explainers.data", "dataset_config.toml"
    ) as toml_path:
        config = load(toml_path)

    for dataset in config:
        data = read_dataset(dataset)
        continuous_features = data["continuous_features"]
        categorical_features = data["categorical_features"]
        features = data["features"]
        target = data["target"]
        preprocessor, target_encoder = create_data_transformer(
            continuous_features,
            categorical_features,
        )

        encoded_features = preprocessor.fit_transform(features)
        encoded_target = target_encoder.fit_transform(target)

        X_train, X_test, y_train, y_test = train_test_split(
            encoded_features,
            encoded_target,
            test_size=0.2,
            random_state=42,
        )
        model = RandomForestClassifier()
        hyperparam_tuner = RandomizedSearchCV(
            estimator=model,
            param_distributions=params["RF"],
            n_iter=100,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )
        hyperparam_tuner.fit(X_train, y_train)
        best_model = hyperparam_tuner.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        result = {
            "dataset": "adult",
            "classifier": "Random Forest",
            "accuracy_train": accuracy_score(y_train, y_pred_train),
            "accuracy_test": accuracy_score(y_test, y_pred_test),
            "f1_macro_train": f1_score(y_train, y_pred_train, average="macro"),
            "f1_macro_test": f1_score(y_test, y_pred_test, average="macro"),
            "f1_micro_train": f1_score(y_train, y_pred_train, average="micro"),
            "f1_micro_test": f1_score(y_test, y_pred_test, average="micro"),
        }
        print(result)


if __name__ == "__main__":
    main()
