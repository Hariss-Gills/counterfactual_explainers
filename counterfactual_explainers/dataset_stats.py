from counterfactual_explainers.data.preprocess_data import create_data_transformer, read_dataset, get_compas_dataset

def get_pipeline_stats(df, target_label, scaler='minmax', encode='onehot'):
    target = df[target_label]
    features = df.drop(columns=[target_label])
    categorical_features = features.select_dtypes(include=["object", "category"]).columns
    continuous_features = features.select_dtypes(exclude=["object", "category"]).columns

    preprocessor, target_encoder = create_data_transformer(continuous_features, categorical_features, scaler, encode)

    num_of_records, num_of_features = features.shape

    num_of_cont = len(continuous_features)
    num_of_cat = len(categorical_features)
    
    #TODO: Define somewhere
    # num_of_act =

    encoded_features = preprocessor.fit_transform(features)
    num_encoded_features = encoded_features.shape[1]

    target_encoder.fit_transform(target)
    num_labels = len(target_encoder.classes_)

    return num_of_records, num_of_features, num_of_cont, num_of_cat, num_encoded_features, num_labels


def main():
    df, target_label = read_dataset("adult")
    stats = get_pipeline_stats(df, target_label)
    print(stats)
    df, target_label = get_compas_dataset()
    stats = get_pipeline_stats(df, target_label)
    print(stats)
    df, target_label = read_dataset("fico")
    stats = get_pipeline_stats(df, target_label)
    print(stats)
    df, target_label = read_dataset("german_credit")
    stats = get_pipeline_stats(df, target_label)
    print(stats)

if __name__ == "__main__":
    main()
