from data.preprocess_data import create_pipeline, get_adult_dataset, get_compas_dataset, get_german_dataset, get_fico_dataset

def get_pipeline_stats(df, target_label, scaler='minmax', encode='onehot'):
    target = df[target_label]
    features = df.drop(columns=[target_label])
    categorical_features = features.select_dtypes(include=["object", "category"]).columns
    continuous_features = features.select_dtypes(exclude=["object", "category"]).columns

    preprocessor, target_encoder = create_pipeline(continuous_features, categorical_features, scaler, encode)

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
    df, target_label = get_adult_dataset("data/raw_data/adult.csv")
    stats = get_pipeline_stats(df, target_label)
    print(stats)
    df, target_label = get_compas_dataset("data/raw_data/compas-scores-two-years.csv")
    stats = get_pipeline_stats(df, target_label)
    print(stats)
    df, target_label = get_fico_dataset("data/raw_data/fico.csv")
    stats = get_pipeline_stats(df, target_label)
    print(stats)
    df, target_label = get_german_dataset("data/raw_data/german_credit.csv")
    stats = get_pipeline_stats(df, target_label)
    print(stats)

if __name__ == "__main__":
    main()
