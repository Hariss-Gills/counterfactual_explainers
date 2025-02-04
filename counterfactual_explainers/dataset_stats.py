from importlib.resources import files
from tomllib import load

from counterfactual_explainers.data.preprocess_data import (
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)


def get_pipeline_stats(data, scaler="minmax", encode="onehot"):
    continuous_features = data["continuous_features"]
    categorical_features = data["categorical_features"]
    features = data["features"]
    target = data["target"]
    non_act_features = data["non_act_features"]
    preprocessor, target_encoder = create_data_transformer(
        continuous_features, categorical_features, scaler, encode
    )

    num_of_records, num_of_features = features.shape

    num_of_cont = len(continuous_features)
    num_of_cat = len(categorical_features)

    num_of_act = num_of_features - len(non_act_features)

    num_encoded_features = None

    if encode:
        encoded_features = preprocessor.fit_transform(features)
        num_encoded_features = encoded_features.shape[1]

    target_encoder.fit_transform(target)
    num_labels = len(target_encoder.classes_)

    return (
        num_of_records,
        num_of_features,
        num_of_cont,
        num_of_cat,
        num_of_act,
        num_encoded_features,
        num_labels,
    )


def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    # TODO: save the stats as a dataframe
    for dataset in config["dataset"]:
        if dataset == "compas":
            data = read_compas_dataset()
        else:
            data = read_dataset(dataset)

        if dataset == "fico":
            stats = get_pipeline_stats(data, encode=None)
        else:
            stats = get_pipeline_stats(data)

        print(stats)


if __name__ == "__main__":
    main()
