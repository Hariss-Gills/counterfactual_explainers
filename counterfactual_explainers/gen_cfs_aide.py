from importlib.resources import files
from pathlib import Path
from tomllib import load

import pandas as pd
from keras.models import load_model as load_keras_model
from sklearn.model_selection import train_test_split

from counterfactual_explainers.aide.aide_explain import init_var_optAINet
from counterfactual_explainers.aide.prepare_data import (
    decode_df,
    get_line_columns,
    get_prob_dict,
)
from counterfactual_explainers.aide.prepare_data import (
    read_adult_dataset as aide_read_adult,
)
from counterfactual_explainers.aide.prepare_data import (
    read_compas_dataset as aide_read_compas,
)
from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    read_compas_dataset,
    read_dataset,
)

results_path = Path("./counterfactual_explainers/results")


# TODO: Metrics like runtime cannot be measured post-hoc
# So this will have to be done here
# NOTE: I have to find out why whenever I increase the pop_size
# the num of cfs always converges to 1. This does not happen in the notebook


def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = clean_config(config)
    results_path.mkdir(parents=True, exist_ok=True)

    for dataset in config["dataset"]:
        if dataset == "compas":
            data = read_compas_dataset()
            aide_data_object = aide_read_compas()

        else:
            data = read_dataset(dataset)
            aide_data_object = aide_read_adult(dataset)

        for model_name in config["model"]:
            if dataset == "german_credit" and model_name == "DNN":

                params_model = config["model"][model_name]
                params_dataset = config["dataset"][dataset]

                seed = params_model["classifier__random_state"][0]

                features = data["features"]
                target = data["target"]

                # HACK: this is needed so the same query_instance
                # is chosen for dice and aide.

                X_train_df, X_test_df, y_train_df, y_test_df = (
                    train_test_split(
                        features,
                        target,
                        test_size=params_dataset["test_size"],
                        random_state=seed,
                        stratify=target,
                    )
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    aide_data_object["X"],
                    aide_data_object["y"],
                    test_size=params_dataset["test_size"],
                    random_state=seed,
                    stratify=aide_data_object["y"],
                )

                query_instance = X_test_df.sample(random_state=0)
                index_in_arr = X_test_df.index.get_loc(query_instance.index[0])
                encoded_query_instance = X_test[index_in_arr]

                encoded_query_instance_df = pd.DataFrame(
                    data=encoded_query_instance.reshape(1, -1),
                    columns=aide_data_object["X_columns_with_dummies"],
                )

                decoded_query_instance_df = decode_df(
                    encoded_query_instance_df, aide_data_object
                )

                # NOTE: This should be the same as dice query_instance
                decoded_query_instance_df.to_csv(
                    results_path
                    / f"cf_aide_{model_name}_{dataset}_query_instance.csv",
                    index=False,
                )
                print(decoded_query_instance_df)

                model = load_keras_model(
                    f"counterfactual_explainers/models/{model_name}"
                    f"_AIDE_{dataset}.keras"
                )

                lime_coeffs_reorder = list()
                df_out = pd.DataFrame(
                    columns=get_line_columns(aide_data_object)
                )
                db_file = results_path / "ignore_AIDE_demo.db"
                prob_dict = get_prob_dict(
                    encoded_query_instance, model, aide_data_object
                )
                for num_required_cfs in range(1, 21):

                    # NOTE: this num_required_cfs is very hard to control
                    # but usually increasing the affinity_constant and pop_size
                    # does the trick maybe I can try to scale affinity_constant since
                    # it has more of an effect.

                    # NOTE: I had to do merge the two classes for compas here
                    # compas 55, 2.50
                    # adult 50, 3.00
                    # fico 50 , 1.00
                    # german 50, 0.1875
                    parameter_dict = {
                        "sort_by": "distance",
                        "use_mads": True,
                        "problem_size": 1,
                        "search_space": [0, 1],
                        "max_gens": 5,
                        "pop_size": 50,  # Let's find best affinity_constant for 50
                        "num_clones": 10,
                        "beta": 1,
                        "num_rand": 2,
                        "affinity_constant": 0.1875,
                        # NOTE: for fico
                        # 0.025 -> 1 was a fail only 2 CFS returned.
                        # 1.5 -> 29 need to go lower
                        # 1.4 -> 30
                        # 1.2 -> 30
                        # 1.05 -> 29
                        # 1.03 -> 29
                        # 1.01 -> 30
                        # 1.00 -> 30
                        # 1.00 -> 30
                        "stop_condition": 0.01,
                        "new_cell_rate": 1.0,
                    }
                    result, df_out = init_var_optAINet(
                        model,
                        X_test,
                        index_in_arr,
                        aide_data_object,
                        prob_dict,
                        db_file,
                        lime_coeffs_reorder,
                        df_out,
                        parameter_dict,
                    )

                    decoded_cfs = decode_df(df_out, aide_data_object)

                    if not df_out.empty:
                        decoded_cfs.to_csv(
                            results_path
                            / f"cf_aide_{model_name}_{dataset}_{num_required_cfs}.csv",
                            index=False,
                        )
                        print(decoded_cfs)
                        print(result)
                    else:
                        print(
                            f"AIDE could not find Counterfactuals"
                            f" for {query_instance}"
                        )


if __name__ == "__main__":
    main()
