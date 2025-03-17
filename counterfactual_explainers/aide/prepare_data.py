import csv
import pickle
import re
from importlib.resources import files
from tomllib import load

import keras
import numpy as np
import pandas as pd

# import prepare_dataset  # from LORE
from keras import datasets
from scipy import stats
from scipy.sparse import isspmatrix_csr
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

from counterfactual_explainers.aide import predict
from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    create_data_transformer,
    read_dataset,
)


def get_prob_dict(encoded_query_instance, model, dataset):
    prob_dict = {}

    bb_input_line = keras.ops.expand_dims(encoded_query_instance, axis=0)
    print(bb_input_line.shape)

    prob = predict.predict_no_reshape(model, bb_input_line)
    prob_array = np.asarray([1 - prob, prob], dtype=float).reshape(1, -1)
    prob_pos = 0
    for outcome in dataset["possible_outcomes"]:
        prob_dict[outcome] = prob_array[0][prob_pos]
        prob_pos = prob_pos + 1

    return prob_dict


def get_line_columns(dataset):
    line_columns = dataset["X_columns"]
    if dataset["use_dummies"] == True:
        line_columns = dataset["X_columns_with_dummies"]
    return line_columns


def decode_df(df_out, dataset):
    decoded_df = df_out.copy()

    for feature in dataset["dummy"]:
        dummy_cols = dataset["dummy"][feature]
        existing_cols = [
            col for col in dummy_cols if col in decoded_df.columns
        ]
        if not existing_cols:
            continue

        decoded_df[feature] = decoded_df[existing_cols].idxmax(axis=1)
        decoded_df[feature] = decoded_df[feature].str.replace(
            f"{feature}_", "", regex=False
        )
        decoded_df.drop(columns=existing_cols, inplace=True)

    if dataset["continuous"]:
        continuous_cols = [
            col for col in dataset["continuous"] if col in decoded_df.columns
        ]
        if continuous_cols:
            scaler = dataset["scaler"][dataset["continuous"][0]]
            scaled_values = decoded_df[continuous_cols].values

            original_values = scaler.inverse_transform(scaled_values)
            decoded_df[continuous_cols] = original_values

    decoded_df = decoded_df.reindex(columns=dataset["X_columns"])

    return decoded_df


# def bucket_mixed_data(dce=False):
#     def fit_bins(values, num_quantiles):
#         qt = KBinsDiscretizer(
#             n_bins=num_quantiles, encode="ordinal", strategy="quantile"
#         )
#         qt = qt.fit(values)
#         return qt
#
#     # path = '~/github/data_exploration/data_exploration/'
#     url = "datasets/heloc_dataset_v1.csv"
#     df = pd.read_csv(url)
#     target = df["RiskPerformance"]
#     df_no_class = df.copy(deep=True)
#     df_no_class = df_no_class.drop(["RiskPerformance"], axis=1)
#     # define columns
#     is_categorical = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]
#     make_categorical = [
#         "ExternalRiskEstimate",
#         "MSinceOldestTradeOpen",
#         "MSinceMostRecentTradeOpen",
#         "AverageMInFile",
#         "NumSatisfactoryTrades",
#         "NumTrades60Ever2DerogPubRec",
#         "NumTrades90Ever2DerogPubRec",
#         "PercentTradesNeverDelq",
#         "MSinceMostRecentDelq",
#         "NumTotalTrades",
#         "NumTradesOpeninLast12M",
#         "PercentInstallTrades",
#         "MSinceMostRecentInqexcl7days",
#         "NumInqLast6M",
#         "NumInqLast6Mexcl7days",
#         "NetFractionRevolvingBurden",
#         "NetFractionInstallBurden",
#         "NumRevolvingTradesWBalance",
#         "NumInstallTradesWBalance",
#         "NumBank2NatlTradesWHighUtilization",
#         "PercentTradesWBalance",
#     ]
#     df_drop_minus9 = df.copy(deep=True)
#     # remove all -9s
#     for row in df_drop_minus9.itertuples():
#         flag = False
#         for i in range(len(df_drop_minus9.columns)):
#             if row[i] == -9:
#                 flag = True
#         if flag == True:
#             index = row[0]
#             df_drop_minus9.drop(labels=index, axis=0, inplace=True)
#     print(df_drop_minus9.shape)
#     df_drop_minus9 = df_drop_minus9.reset_index()
#     binner = {}  # dicts insted of list of dicts
#
#     def get_df_dce(df_in):
#         df_dce = df_in.copy(deep=True)
#         df_dce.drop(columns="index", inplace=True)
#         return df_dce
#
#     if dce == True:
#         df_dce = get_df_dce(df_drop_minus9)
#     # loop through all categorical
#     for col in make_categorical:
#         # bin_dict = {}
#         values = []
#         for i in range(df_drop_minus9.shape[0]):
#             val = df_drop_minus9[col][i]
#             if val >= 0:
#                 values.append(val)
#         # fit QuantileTransformer()
#         values = np.array(values).reshape(-1, 1)
#         qt = fit_bins(values, 10)
#         # bin_dict[col] = qt
#         binner[col] = qt
#         # binner.append(bin_dict)
#         # change df column to np array
#         col_array = df_drop_minus9[col].values
#         new_list = []
#         for i in range(len(col_array)):
#             if col_array[i] >= 0:
#                 new_val = qt.transform([[col_array[i]]])
#                 new_list.append(new_val[0][0])
#                 # import pdb; pdb.set_trace()
#             else:
#                 new_list.append(col_array[i])
#         df_drop_minus9[col] = pd.Series(data=new_list, name=col)
#     # remove unnecessary index
#     df_drop_minus9.drop(columns="index", inplace=True)
#     csv_columns = df_drop_minus9.columns
#     csv_file = "datasets/fico_heloc_categorical.csv"
#     df_drop_minus9.to_csv(path_or_buf=csv_file, columns=csv_columns)
#     if dce == True:
#         return df_drop_minus9, is_categorical, make_categorical, binner, df_dce
#     else:
#         return df_drop_minus9, is_categorical, make_categorical, binner
#
#
# def label_encode(df, columns, label_encoder=None):
#     df_le = df.copy(deep=True)
#     new_le = label_encoder is None
#     label_encoder = dict() if new_le else label_encoder
#     for col in columns:
#         if new_le:
#             le = LabelEncoder()
#             df_le[col] = le.fit_transform(df_le[col])
#             label_encoder[col] = le
#         else:
#             le = label_encoder[col]
#             df_le[col] = le.transform(df_le[col])
#     return df_le, label_encoder
#
#
# # drops all rows where a given value appear (originally a -9)
# def drop_value(df, value):
#     df_drop = df.copy(deep=True)
#     for row in df_drop.itertuples():
#         flag = False
#         for i in range(len(df_drop.columns)):
#             if row[i] == value:
#                 flag = True
#         if flag == True:
#             index = row[0]
#             df_drop.drop(labels=index, axis=0, inplace=True)
#     # print(df_drop_minus9.shape)
#     df_drop = df_drop.reset_index()
#     return df_drop
#
#
# def bin_columns(df, number_of_bins, columns, binner=None):
#     strategy = "quantile"
#     encoder = "ordinal"
#     df_bin = df.copy(deep=True)
#     # loop through all categorical
#     new_bin = binner is None
#     binner = dict() if new_bin else binner
#     for col in columns:
#         if new_bin:
#             bin = KBinsDiscretizer(
#                 n_bins=number_of_bins, encode=encoder, strategy=strategy
#             )
#             df_bin[col] = bin.fit_transform(df_bin[[col]])
#             binner[col] = bin
#         else:
#             bin = KBinsDiscretizer(
#                 n_bins=number_of_bins, encode=encoder, strategy=strategy
#             )
#             df_bin[col] = bin.transform(df_bin[[col]])
#     return df_bin, binner
#
#
def get_possible_outcomes(df, class_name):
    possible_outcomes = list(df[class_name].unique())
    p0 = possible_outcomes[0]
    p1 = possible_outcomes[1]
    labels = possible_outcomes[0], possible_outcomes[1]
    # labels = 0,1
    df[class_name].replace(p0, 0, inplace=True)
    df[class_name].replace(p1, 1, inplace=True)
    return df, possible_outcomes, labels


#
# def process_lending():
#     path = "datasets/"
#     file_name = "LoanStats3a.csv"  #'vate '
#     url = path + file_name
#     df = pd.read_csv(url)
#     selected_columns = [
#         "loan_status",
#         "emp_length",
#         "home_ownership",
#         "annual_inc",
#         "addr_state",
#         "open_acc",
#         "purpose",
#         "grade",
#         "earliest_cr_line",
#         "loan_amnt",
#         "term",
#     ]  # loan_amnt & termadded JF 2/8/20
#     X_columns = [
#         "emp_length",
#         "home_ownership",
#         "annual_inc",
#         "open_acc",
#         "purpose",
#         "grade",
#         "loan_amnt",
#         "term",
#     ]  #'earliest_cr_line' assumed to be credit history may be 'mths_since_last_delinq' instead alos removed 'years_of_credit','addr_state',
#     y_column = "loan_status"
#     class_name = y_column
#     non_continuous = [
#         "home_ownership",
#         "purpose",
#         "grade",
#         "term",
#     ]  # removed 'addr_state',
#     continuous = [
#         "emp_length",
#         "annual_inc",
#         "open_acc",
#         "loan_amnt",
#     ]  # removed 'years_of_credit',
#     df = df[selected_columns]  # not all columns wanted
#     df = df.dropna(axis=0)
#     df = df.reset_index()
#     years_of_credit = list()
#     for i in range(
#         df.shape[0]
#     ):  # change dates from month-year to number of years before  September 2011
#         # change feature to a whole number of years of credit history
#         temp_str = str(df["earliest_cr_line"][i])
#         split_strings = temp_str.split("-")
#         years = 2011 - (int(split_strings[1]))
#         if split_strings[0] == "Sep" or "Oct" or "Nov" or "Dec":
#             years = years - 1
#         years_of_credit.append(years)
#     # change emp_length to an int 0-9 for number of years and 10 for 10+ years
#     emp_length_int = list()
#     for i in range(df.shape[0]):
#         value = df["emp_length"][i]
#         out = 11
#         if value == "< 1 year":
#             out = 0
#         elif value == "1 year":
#             out = 1
#         elif value == "2 years":
#             out = 2
#         elif value == "3 years":
#             out = 3
#         elif value == "4 years":
#             out = 4
#         elif value == "5 years":
#             out = 5
#         elif value == "6 years":
#             out = 6
#         elif value == "7 years":
#             out = 7
#         elif value == "8 years":
#             out = 8
#         elif value == "9 years":
#             out = 9
#         elif value == "10+ years":
#             out = 10
#         if out == 11:
#             print(
#                 "unaccounted for value ",
#                 value,
#                 " in determining employment length",
#             )
#             breakpoint()
#         emp_length_int.append(out)
#     df = df.drop(labels=["earliest_cr_line", "index", "emp_length"], axis=1)
#     df["years_of_credit"] = years_of_credit
#     df["emp_length"] = emp_length_int
#
#     df_X = df[X_columns]
#     target = np.empty(df["loan_status"].shape)
#     # below code will conflate 'Fully Paid' and 'Charged Off' with there 'Does not mmet the credit policy. Status' versions cannot be undone
#     for i in range(len(df["loan_status"])):
#         if df["loan_status"][i] == (
#             "Fully Paid"
#             or "Does not meet the credit policy. Status:Fully Paid"
#         ):
#             df["loan_status"][i] = "Fully Paid"
#             target[i] = 1
#         else:
#             df["loan_status"][i] = "Charged Off"
#             target[i] = 0
#
#     def get_df_dce(df_in):
#         df_dce = df_in.copy(deep=True)
#         # df_dce.drop(columns = 'index',inplace=True)
#         return df_dce
#
#     df_dce = get_df_dce(
#         df
#     )  # need a unenencoded/scaled dataset for dcf and dce moved below loan status to int
#     # df_dce still has arr_state and years_of credit, if used again will need removing
#     invariants = []
#     binners = []
#     label_encoders = []
#     use_dummies = True
#     # gives labels in wrong order with fully paid as 0 and and and charged off as 1 hard code instead
#     # df,possible_outcomes,labels = get_possible_outcomes(df,class_name)
#     possible_outcomes = ["Charged Off", "Fully Paid"]  # must be in order [0,1]
#     labels = ("Charged Off", "Fully Paid")  # must be in order [0,1]
#     type_features, features_type = prepare_dataset.recognize_features_type(
#         df, class_name
#     )
#     idx_features = {i: col for i, col in enumerate(X_columns)}
#     dummy = {}
#     scalers = {}
#     mads = (
#         {}
#     )  # for use with lime as a fiddle factor to allow lime coeffs to compare continuous to non_continuous , add dict of continuous variables {variable : mad}
#     for column in df_X.columns:
#         if column not in non_continuous:  # for continuous columns
#
#             # scaling when using dice which requires normalization for compatability
#             scaler = MinMaxScaler()
#             values = df_X[column].values.reshape(-1, 1)
#             mads[column] = stats.median_absolute_deviation(values, axis=None)
#             scaler.fit(values)
#             df_X[column] = scaler.transform(values)
#             scalers[column] = scaler
#
#             """
#             #change to standardising continuous columns, use this when not using dice
#             #scaler = StandardScaler()#used at start of big_survey
#             scaler = StandardScaler()#defaults to range (0,1) this is sclaer for trying on test of big survey
#             values = (df_X[column].values.reshape(-1,1))
#             mads[column] = stats.median_absolute_deviation(values, axis = None)
#             scaler.fit(values)
#             df_X[column] = scaler.transform(values)
#             scalers[column] = scaler
#             """
#
#         else:  # for categorical columns
#             # use get dummies
#             dummy_columns = pd.get_dummies(
#                 df_X[column], prefix=column, prefix_sep="_", drop_first=False
#             )
#             for col in dummy_columns.columns:
#                 df_X[col] = dummy_columns[col]
#             update_dict = {column: dummy_columns.columns.values}
#             dummy.update(update_dict)
#             # encoder = LabelEncoder()
#             # encoder = OneHotEncoder()
#             # encoder.fit(df_X[column].values.reshape(-1,1))
#             df_X.drop(column, axis=1, inplace=True)
#
#     dataset = {
#         "name": file_name.replace(".csv", ""),
#         "df": df,  # removed to shrink size of dataset obj
#         "df_dce": df_dce,  # removed to shrink size of dataset obj
#         "columns": df.columns,
#         "X_columns": X_columns,
#         "X_columns_with_dummies": df_X.columns,
#         "class_name": class_name,
#         "possible_outcomes": possible_outcomes,
#         "type_features": type_features,
#         "features_type": features_type,
#         #'discrete': discrete,
#         "continuous": continuous,
#         #'categorical': categorical, #added by JF
#         "non_continuous": non_continuous,
#         "discrete": non_continuous,
#         "dummy": dummy,  # added by JF
#         "idx_features": idx_features,
#         "label_encoder": label_encoders,
#         "scaler": scalers,
#         "binner": binners,
#         "number_of_bins": 10,
#         "mads": mads,
#         "labels": labels,
#         "invariants": invariants,
#         "data_human_dict": get_lending_human_dictionary(),
#         "human_data_dict": get_lending_data_dictionary(),
#         "X": df_X.values,  # np array #removed to shrink size of dataset obj
#         "y": target,  # np array #removed to shrink size of dataset obj
#         "use_dummies": use_dummies,
#     }
#     pickle_filename = "pickled_data/lending_pickled_data_MinMax_27_06_22.p"
#     outfile = open(pickle_filename, "wb")
#     pickle.dump(dataset, outfile)
#     outfile.close()
#     return dataset
#
#
# def get_lending_data_dictionary():
#     d = {}
#     d["value of loan"] = "loan_amnt"
#     d["term of loan"] = "term"
#     d["length of employment in years"] = "emp_length"
#     d["annual income"] = "annual_inc"
#     d["number of open accounts"] = "open_acc"
#     d["home ownership"] = "home_ownership"
#     d["purpose"] = "purpose"
#     d["grade"] = "grade"
#     return d
#
#
# def get_lending_human_dictionary():
#     d = {}  # init empty dict
#     d["loan_amnt"] = "value of loan"
#     d["term_ 60 months"] = {"name": "term of loan", "value": "60 months"}
#     d["term_ 36 months"] = {"name": "term of loan", "value": "36 months"}
#     d["emp_length"] = "length of employment in years"
#     d["annual_inc"] = "annual income"
#     d["open_acc"] = "number of open accounts"
#     d["years_of_credit"] = "number of years of credit history"
#     d["home_ownership_MORTGAGE"] = {
#         "name": "home ownership",
#         "value": "mortgaged",
#     }
#     d["home_ownership_OTHER"] = {"name": "home ownership", "value": "other"}
#     d["home_ownership_RENT"] = {"name": "home ownership", "value": "rent"}
#     d["home_ownership_OWN"] = {"name": "home ownership", "value": "own"}
#     d["home_ownership_NONE"] = {"name": "home ownership", "value": "none"}
#     """
#     #state imnformation not used any more
#     d['addr_state_AK'] = {'name':'U.S. state of residence','value':'Alaska'}
#     d['addr_state_AL'] = {'name':'U.S. state of residence','value':'Alabama'}
#     d['addr_state_AR'] = {'name':'U.S. state of residence','value':'Arkansas'}
#     d['addr_state_AZ'] = {'name':'U.S. state of residence','value':'Arizona'}
#     d['addr_state_CA'] = {'name':'U.S. state of residence','value':'California'}
#     d['addr_state_CO'] = {'name':'U.S. state of residence','value':'Colorado'}
#     d['addr_state_CT'] = {'name':'U.S. state of residence','value':'Connecticut'}
#     d['addr_state_DC'] = {'name':'U.S. state of residence','value':'District of Columbia'}
#     d['addr_state_DE'] = {'name':'U.S. state of residence','value':'Delaware'}
#     d['addr_state_FL'] = {'name':'U.S. state of residence','value':'Florida'}
#     d['addr_state_GA'] = {'name':'U.S. state of residence','value':'Georgia'}
#     d['addr_state_HI'] = {'name':'U.S. state of residence','value':'Hawaii'}
#     d['addr_state_IA'] = {'name':'U.S. state of residence','value':'Iowa'}
#     d['addr_state_ID'] = {'name':'U.S. state of residence','value':'Idaho'}
#     d['addr_state_IL'] = {'name':'U.S. state of residence','value':'Illinois'}
#     d['addr_state_IN'] = {'name':'U.S. state of residence','value':'Indiana'}
#     d['addr_state_KS'] = {'name':'U.S. state of residence','value':'Kansas'}
#     d['addr_state_KY'] = {'name':'U.S. state of residence','value':'Kentucky'}
#     d['addr_state_LA'] = {'name':'U.S. state of residence','value':'Louisiana'}
#     d['addr_state_MA'] = {'name':'U.S. state of residence','value':'Massachusetts'}
#     d['addr_state_MD'] = {'name':'U.S. state of residence','value':'Maryland'}
#     d['addr_state_ME'] = {'name':'U.S. state of residence','value':'Maine'}
#     d['addr_state_MI'] = {'name':'U.S. state of residence','value':'Michigan'}
#     d['addr_state_MN'] = {'name':'U.S. state of residence','value':'Minnesota'}
#     d['addr_state_MO'] = {'name':'U.S. state of residence','value':'Missouri'}
#     d['addr_state_MS'] = {'name':'U.S. state of residence','value':'Mississippi'}
#     d['addr_state_MT'] = {'name':'U.S. state of residence','value':'Montana'}
#     d['addr_state_NC'] = {'name':'U.S. state of residence','value':'North Carolina'}
#     d['addr_state_NE'] = {'name':'U.S. state of residence','value':'Nebraska'}
#     d['addr_state_NH'] = {'name':'U.S. state of residence','value':'New Hampshire'}
#     d['addr_state_NJ'] = {'name':'U.S. state of residence','value':'New Jersey'}
#     d['addr_state_NM'] = {'name':'U.S. state of residence','value':'New Mexico'}
#     d['addr_state_NV'] = {'name':'U.S. state of residence','value':'Nevada'}
#     d['addr_state_NY'] = {'name':'U.S. state of residence','value':'New York'}
#     d['addr_state_OH'] = {'name':'U.S. state of residence','value':'Ohio'}
#     d['addr_state_OK'] = {'name':'U.S. state of residence','value':'Oklahoma'}
#     d['addr_state_OR'] = {'name':'U.S. state of residence','value':'Oregon'}
#     d['addr_state_PA'] = {'name':'U.S. state of residence','value':'Pennsylvania'}
#     d['addr_state_RI'] = {'name':'U.S. state of residence','value':'Rhode Island'}
#     d['addr_state_SC'] = {'name':'U.S. state of residence','value':'South Carolina'}
#     d['addr_state_SD'] = {'name':'U.S. state of residence','value':'South Dakota'}
#     d['addr_state_TN'] = {'name':'U.S. state of residence','value':'Tennessee'}
#     d['addr_state_TX'] = {'name':'U.S. state of residence','value':'Texas'}
#     d['addr_state_UT'] = {'name':'U.S. state of residence','value':'Utah'}
#     d['addr_state_VA'] = {'name':'U.S. state of residence','value':'Viginia'}
#     d['addr_state_VT'] = {'name':'U.S. state of residence','value':'Vermont'}
#     d['addr_state_WA'] = {'name':'U.S. state of residence','value':'Washington'}
#     d['addr_state_WI'] = {'name':'U.S. state of residence','value':'Wisconsin'}
#     d['addr_state_WV'] = {'name':'U.S. state of residence','value':'West Virginia'}
#     d['addr_state_WY'] = {'name':'U.S. state of residence','value':'Wyoming'}
#     """
#     d["purpose_car"] = {"name": "purpose of loan", "value": "car"}
#     d["purpose_credit_card"] = {
#         "name": "purpose of loan",
#         "value": "credit card",
#     }
#     d["purpose_debt_consolidation"] = {
#         "name": "purpose of loan",
#         "value": "debt consolidation",
#     }
#     d["purpose_educational"] = {
#         "name": "purpose of loan",
#         "value": "educational",
#     }
#     d["purpose_home_improvement"] = {
#         "name": "purpose of loan",
#         "value": "home improvement",
#     }
#     d["purpose_house"] = {"name": "purpose of loan", "value": "house"}
#     d["purpose_major_purchase"] = {
#         "name": "purpose of loan",
#         "value": "major purchase",
#     }
#     d["purpose_medical"] = {"name": "purpose of loan", "value": "medical"}
#     d["purpose_moving"] = {"name": "purpose of loan", "value": "moving"}
#     d["purpose_other"] = {"name": "purpose of loan", "value": "other"}
#     d["purpose_renewable_energy"] = {
#         "name": "purpose of loan",
#         "value": "renewable energy",
#     }
#     d["purpose_small_business"] = {
#         "name": "purpose of loan",
#         "value": "small business",
#     }
#     d["purpose_vacation"] = {"name": "purpose of loan", "value": "vacation"}
#     d["purpose_wedding"] = {"name": "purpose of loan", "value": "wedding"}
#     d["grade_A"] = {"name": "credit rating grade", "value": "A"}
#     d["grade_B"] = {"name": "credit rating grade", "value": "B"}
#     d["grade_C"] = {"name": "credit rating grade", "value": "C"}
#     d["grade_D"] = {"name": "credit rating grade", "value": "D"}
#     d["grade_E"] = {"name": "credit rating grade", "value": "E"}
#     d["grade_F"] = {"name": "credit rating grade", "value": "F"}
#     d["grade_G"] = {"name": "credit rating grade", "value": "G"}
#     return d
#
#
# # loads porcesses by romoving all rows with a value in (-9 in the original case)
# # then saves the df and returna dataset dict object
# def process_fico_data():  # replaces process_cat_dataset with what worked in data_exploration/data_exploration.py
#     # df has had all '-9' values dropped, all +ve value binned to deciles
#     # is_categorical the two columns already categorical
#     # make categorical the remaining columns that have been binned in bucket_mixed_data()
#     # binner list of dicts k = column name, v =  fitted KBinsDiscretizer object
#     # if seperate df of dce data is required dce = True else False
#     dce = True
#     if dce == True:
#         df, is_categorical, make_categorical, binner, df_dce = (
#             bucket_mixed_data(dce)
#         )
#     else:
#         df, is_categorical, make_categorical, binner = bucket_mixed_data(dce)
#         df_dce = "no dce dataset"
#     class_name = "RiskPerformance"
#     file_name = "heloc_dataset_v1.csv"
#     X_columns = list(df.columns.copy(deep=True))
#     X_columns.remove(class_name)
#     df_encoded, label_encoder = label_encode(df, X_columns)
#
#     df_encoded, possible_outcomes, labels = get_possible_outcomes(
#         df_encoded, class_name
#     )
#     invariants = ""
#     scaler = ""
#     # df_encoded = pd.get_dummies(df_encoded,make_categorical)
#     X = df_encoded[X_columns].values
#     y = df_encoded[class_name].values
#
#     type_features, features_type = prepare_dataset.recognize_features_type(
#         df, class_name
#     )
#     idx_features = {i: col for i, col in enumerate(X_columns)}
#
#     fico_human_dict = {
#         "RiskPerformance": "Decision on awarding your application",
#         "ExternalRiskEstimate": "External estimated score for your application",
#         "MSinceOldestTradeOpen": "Age of your oldest credit in months",
#         "MSinceMostRecentTradeOpen": "Age of your newest credit in months",
#         "AverageMInFile": "Average age of your credits in months",
#         "NumSatisfactoryTrades": "Number of credits you have repaid in full and on time",
#         "NumTrades60Ever2DerogPubRec": "Number of credits you have repaid 60 days late",
#         "NumTrades90Ever2DerogPubRec": "Number of credits you have repaid 90 days late",
#         "PercentTradesNeverDelq": "Percentage of your credits with no late payments",
#         "MSinceMostRecentDelq": "Months since your most recent late payment",
#         "MaxDelq2PublicRecLast12M": "Latest payment on your credits in public records in the last 12 Months",
#         "MaxDelqEver": "Latest payment on your credits",
#         "NumTotalTrades": "Your total number of credits ever",
#         "NumTradesOpeninLast12M": "Number of credits taken by you in last 12 months",
#         "PercentInstallTrades": "Percentage of your credits paid by installment",
#         "MSinceMostRecentInqexcl7days": "Months since last inquiry (excluding last 7 days) about credit for you",
#         "NumInqLast6M": "Total number of inquiries in last 6 months about credit for you",
#         "NumInqLast6Mexcl7days": "Total number of inquiries in last 6 months (excluding last 7 days) about credit for you",
#         "NetFractionRevolvingBurden": "Balance of your credit cards, divided by your credit limit, as a percentage",
#         "NetFractionInstallBurden": "Your installment balance divided by your original credit amount as a percentage",
#         "NumRevolvingTradesWBalance": "Your number of credit cards with a balance still to pay",
#         "NumInstallTradesWBalance": "Your number of installment credits with a balance still to pay",
#         "NumBank2NatlTradesWHighUtilization": "The number of your credits near to their limit",
#         "PercentTradesWBalance": "Percentage of your credits with a balance to pay",
#     }  # must be kept synchronised with write_db add attributes bundle or even get both to same place
#     human_fico_dict = {}
#     for key, value in fico_human_dict.items():
#         human_fico_dict[value] = key
#     dataset = {
#         "name": file_name.replace(".csv", ""),
#         "df": df_encoded,
#         "columns": df_encoded.columns,
#         "X_columns": X_columns,
#         "class_name": class_name,
#         "possible_outcomes": possible_outcomes,
#         "type_features": type_features,
#         "features_type": features_type,
#         #'discrete': discrete,
#         "continuous": "",
#         #'categorical': categorical, #added by JF
#         "non_continuous": X_columns,
#         "discrete": df.columns,
#         #'dummy': dummy, #added by JF
#         "idx_features": idx_features,
#         "label_encoder": label_encoder,
#         "scaler": scaler,
#         "binner": binner,
#         "number_of_bins": 10,
#         "labels": labels,
#         "invariants": invariants,
#         "fico_human_dict": fico_human_dict,
#         "human_fico_dict": human_fico_dict,
#         #'permitted_neg_numbers':{-7,-8,-9},
#         #'datatypes' : datatypes,
#         #'positive_attributes' : positive_attributes,
#         "X": X,  # encoded np array
#         "y": y,  # encoded np array
#         "df_dce": df_dce,
#     }
#     pickle_filename = "pickled_data/pickled_data.p"
#     outfile = open(pickle_filename, "wb")
#     pickle.dump(dataset, outfile)
#     outfile.close()
#
#     return dataset
#
#
# def process_cat_dataset():
#     class_name = "RiskPerformance"
#     filename = "heloc_dataset_v1.csv"
#     url = "datasets/" + filename
#     df = pd.read_csv(url)
#     target = df[class_name]
#     df_no_class = df.copy(deep=True)
#     df_no_class = df_no_class.drop(["RiskPerformance"], axis=1)
#     # define columns
#     is_categorical = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]
#     make_categorical = [
#         "ExternalRiskEstimate",
#         "MSinceOldestTradeOpen",
#         "MSinceMostRecentTradeOpen",
#         "AverageMInFile",
#         "NumSatisfactoryTrades",
#         "NumTrades60Ever2DerogPubRec",
#         "NumTrades90Ever2DerogPubRec",
#         "PercentTradesNeverDelq",
#         "MSinceMostRecentDelq",
#         "NumTotalTrades",
#         "NumTradesOpeninLast12M",
#         "PercentInstallTrades",
#         "MSinceMostRecentInqexcl7days",
#         "NumInqLast6M",
#         "NumInqLast6Mexcl7days",
#         "NetFractionRevolvingBurden",
#         "NetFractionInstallBurden",
#         "NumRevolvingTradesWBalance",
#         "NumInstallTradesWBalance",
#         "NumBank2NatlTradesWHighUtilization",
#         "PercentTradesWBalance",
#     ]
#     df_drop_val = drop_value(df, -9)
#     df_bin, binner = bin_columns(df_drop_val, 10, make_categorical)
#     df_bin.drop(columns="index", inplace=True)
#     # remove unnecessary index
#     csv_columns = df_bin.columns
#     csv_file = "datasets/fico_heloc_categorical.csv"
#     df_bin.to_csv(path_or_buf=csv_file, columns=csv_columns)
#     df_encoded, label_encoder = label_encode(
#         df_bin, make_categorical + is_categorical
#     )
#
#     possible_outcomes = list(df[class_name].unique())
#     p0 = possible_outcomes[0]
#     p1 = possible_outcomes[1]
#     labels = possible_outcomes[0], possible_outcomes[1]
#     # labels = 0,1
#     df_encoded[class_name].replace(p0, 0, inplace=True)
#     df_encoded[class_name].replace(p1, 1, inplace=True)
#     invariants = [""]
#     scalers = [""]
#     # df_encoded = pd.get_dummies(df_encoded,make_categorical)
#     X = df_encoded[df_no_class.columns].values
#     y = df_encoded[class_name].values
#     X_columns = list(df_encoded.columns.copy(deep=True))
#     X_columns.remove("RiskPerformance")
#     type_features, features_type = prepare_dataset.recognize_features_type(
#         df, class_name
#     )
#     idx_features = {i: col for i, col in enumerate(X_columns)}
#     dataset = {
#         "name": filename.replace(".csv", ""),
#         "df": df_encoded,
#         "columns": df_encoded.columns,
#         "X_columns": X_columns,
#         "class_name": "RiskPerformance",
#         "possible_outcomes": possible_outcomes,
#         "type_features": type_features,
#         "features_type": features_type,
#         #'discrete': discrete,
#         "continuous": "",
#         #'categorical': categorical, #added by JF
#         "non_continuous": df_no_class.columns,
#         "discrete": df.columns,
#         #'dummy': dummy, #added by JF
#         "idx_features": idx_features,
#         "label_encoder": label_encoder,
#         "scaler": scalers,
#         "binner": binner,
#         "number_of_bins": 10,
#         "labels": labels,
#         "invariants": invariants,
#         #'permitted_neg_numbers':{-7,-8,-9},
#         #'datatypes' : datatypes,
#         #'positive_attributes' : positive_attributes,
#         "X": X,  # encoded np array
#         "y": y,  # encoded np array
#     }
#
#     return dataset
#
#
# def process_num_dataset():
#     # no drop special value '-9' , no bin no encoding leave to keras
#     class_name = "RiskPerformance"
#     filename = "heloc_dataset_v1.csv"
#     url = "datasets/" + filename
#     df = pd.read_csv(url)
#     target = df[class_name]
#     df_no_class = df.copy(deep=True)
#     df_no_class = df_no_class.drop(["RiskPerformance"], axis=1)
#     # define columns
#     is_categorical = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]
#     make_categorical = [
#         "ExternalRiskEstimate",
#         "MSinceOldestTradeOpen",
#         "MSinceMostRecentTradeOpen",
#         "AverageMInFile",
#         "NumSatisfactoryTrades",
#         "NumTrades60Ever2DerogPubRec",
#         "NumTrades90Ever2DerogPubRec",
#         "PercentTradesNeverDelq",
#         "MSinceMostRecentDelq",
#         "NumTotalTrades",
#         "NumTradesOpeninLast12M",
#         "PercentInstallTrades",
#         "MSinceMostRecentInqexcl7days",
#         "NumInqLast6M",
#         "NumInqLast6Mexcl7days",
#         "NetFractionRevolvingBurden",
#         "NetFractionInstallBurden",
#         "NumRevolvingTradesWBalance",
#         "NumInstallTradesWBalance",
#         "NumBank2NatlTradesWHighUtilization",
#         "PercentTradesWBalance",
#     ]
#     # df_drop_val = drop_value(df,-9)
#     # df_bin, binner = bin_columns(df_drop_val,10,make_categorical)
#     # df_bin.drop(columns = 'index',inplace=True)
#     # remove unnecessary index
#     csv_columns = df.columns
#     csv_file = "datasets/fico_heloc_numerical.csv"
#     df.to_csv(path_or_buf=csv_file, columns=csv_columns)
#     # df_encoded, label_encoder = label_encode(df_bin, make_categorical )
#
#     possible_outcomes = list(df[class_name].unique())
#     p0 = possible_outcomes[0]
#     p1 = possible_outcomes[1]
#     labels = possible_outcomes[0], possible_outcomes[1]
#     # labels = 0,1
#     df[class_name].replace(p0, 0, inplace=True)
#     df[class_name].replace(p1, 1, inplace=True)
#
#     # df_encoded = pd.get_dummies(df_encoded,make_categorical)
#     X = df[df_no_class.columns].values
#     y = df[class_name].values
#     dataset = {
#         "name": filename.replace(".csv", ""),
#         "df": df,
#         "columns": df.columns,
#         "class_name": "RiskPerformance",
#         "possible_outcomes": possible_outcomes,
#         #'type_features': type_features,
#         #'features_type': features_type,
#         #'discrete': discrete,
#         "continuous": "",
#         #'categorical': categorical, #added by JF
#         "non_continuous": df_no_class.columns,
#         #'dummy': dummy, #added by JF
#         #'idx_features': idx_features,
#         "label_encoder": "",
#         #'scaler':scaler,
#         "binner": "",
#         "number_of_bins": 0,
#         "labels": labels,
#         #'permitted_neg_numbers':{-7,-8,-9},
#         #'datatypes' : datatypes,
#         #'positive_attributes' : positive_attributes,
#         "X": X,  # encoded np array
#         "y": y,  # encoded np array
#     }
#     return dataset
#


def get_possible_outcomes(df, class_name):
    possible_outcomes = list(df[class_name].unique())
    p0 = possible_outcomes[0]
    p1 = possible_outcomes[1]
    labels = possible_outcomes[0], possible_outcomes[1]
    # labels = 0,1
    df[class_name].replace(p0, 0, inplace=True)
    df[class_name].replace(p1, 1, inplace=True)
    return df, possible_outcomes, labels


def read_adult_dataset(dataset_name):
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    if dataset_name not in config["dataset"]:
        raise ValueError(
            f"Dataset configuration for '{dataset_name}' not found."
        )

    dataset_config = config["dataset"][dataset_name]
    drop_columns = dataset_config.get("drop_columns", [])
    na_values = dataset_config.get("na_values", None)
    target_label = dataset_config.get("target_name")
    class_name = target_label
    non_act_features = dataset_config.get("non_act_cols")

    package = files("counterfactual_explainers.data.raw_data")
    csv_path = package / f"{dataset_name}.csv"
    with csv_path.open("rb") as file:
        df = pd.read_csv(file, skipinitialspace=True, na_values=na_values)

    df.drop(columns=drop_columns, errors="ignore", inplace=True)

    target = df[target_label]
    features = df.drop(columns=[target_label])
    X_columns = features.columns.tolist()
    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns
    continuous_features = features.select_dtypes(
        exclude=["object", "category"]
    ).columns

    continuous = continuous_features.tolist()
    non_continuous = categorical_features.tolist()

    print(continuous)
    print(non_continuous)

    preprocessor, target_encoder = create_data_transformer(
        continuous_features, categorical_features, "minmax", "onehot"
    )

    preprocessor.fit(features)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps[
        "imputer"
    ]
    cat_imputer = preprocessor.named_transformers_["categorical"].named_steps[
        "imputer"
    ]

    # HACK: this is dirty but
    # does the job

    new_feat_cont = cont_imputer.fit_transform(features[continuous_features])
    new_feat_cont = pd.DataFrame(
        new_feat_cont, columns=continuous_features, index=features.index
    )
    if dataset_name != "fico":
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

    df_dce = pd.concat([features, target], axis=1)
    df_X = df_dce[X_columns]

    print(f"NaN values: {features.isnull().values.any()}")

    transformed_target = target_encoder.fit_transform(target)
    possible_outcomes = target_encoder.classes_.tolist()
    labels = tuple(possible_outcomes)
    target = pd.DataFrame(
        transformed_target, columns=[target.name], index=target.index
    )

    type_features, features_type = recognize_features_type(df, class_name)
    df = pd.concat([features, target], axis=1)

    invariants = non_act_features
    binners = []
    label_encoders = []
    use_dummies = True

    continuous_transformer = preprocessor.named_transformers_["continuous"]
    transformed_continuous = continuous_transformer.transform(
        features[continuous_features]
    )

    transformed_df = pd.DataFrame(
        transformed_continuous,
        columns=continuous_features,
        index=features.index,
    )

    scaler = continuous_transformer.named_steps["scaler"]

    idx_features = {i: col for i, col in enumerate(X_columns)}
    mads = {}
    scalers = {}
    for column in continuous_features:
        mad = stats.median_abs_deviation(
            transformed_df[column], nan_policy="omit"
        )
        mads[column] = mad
        scalers[column] = scaler
    # TODO: somehow get dummy dict

    transformed_array = preprocessor.transform(features)
    if isspmatrix_csr(transformed_array):
        transformed_array = transformed_array.toarray()

    feature_names = preprocessor.get_feature_names_out()
    df_X = pd.DataFrame(
        transformed_array, columns=feature_names, index=features.index
    )
    df_X.columns = df_X.columns.str.replace(
        r"^(continuous__|categorical__)", "", regex=True
    )

    # ohe = preprocessor.named_transformers_["categorical"].named_steps[
    # "encoder"
    # ]

    feature_names = np.array(
        [
            re.sub(r"^(continuous__|categorical__)", "", name)
            for name in feature_names
        ]
    )
    dummy = {}
    for column in non_continuous:
        mask = np.array([fn.startswith(column) for fn in feature_names])
        dummy[column] = feature_names[mask]

    dataset = {
        "name": dataset_name,
        "df": df,  # removed to shrink size of dataset obj
        "df_dce": df_dce,  # removed to shrink size of dataset obj
        "columns": df.columns,
        "X_columns": X_columns,
        "X_columns_with_dummies": df_X.columns,
        "class_name": class_name,
        "possible_outcomes": possible_outcomes,
        "type_features": type_features,
        "features_type": features_type,
        #'discrete': discrete,
        "continuous": continuous,
        #'categorical': categorical, #added by JF
        "non_continuous": non_continuous,
        "discrete": non_continuous,
        "dummy": dummy,  # added by JF
        "idx_features": idx_features,
        "label_encoder": label_encoders,
        "scaler": scalers,
        "binner": binners,
        "number_of_bins": 10,
        "mads": mads,
        "labels": labels,
        "invariants": invariants,
        "data_human_dict": {
            "income": "income",
            "age": "age",
            "work_class_ ?": {"name": "work_class", "value": "?"},
            "work_class_ Federal-gov": {
                "name": "work_class",
                "value": "Federal-gov",
            },
            "work_class_ Local-gov": {
                "name": "work_class",
                "value": "Local-gov",
            },
            "work_class_ Never-worked": {
                "name": "work_class",
                "value": "Never-worked",
            },
            "work_class_ Private": {"name": "work_class", "value": "Private"},
            "work_class_ Self-emp-inc": {
                "name": "work_class",
                "value": "Self-emp-inc",
            },
            "work_class_ Self-emp-not-inc": {
                "name": "work_class",
                "value": "Self-emp-not-inc",
            },
            "work_class_ State-gov": {
                "name": "work_class",
                "value": "State-gov",
            },
            "work_class_ Without-pay": {
                "name": "work_class",
                "value": "Without-pay",
            },
            "education_years": "education_years",
            "marital_status_ Divorced": {
                "name": "marital_status",
                "value": "Divorced",
            },
            "marital_status_ Married-AF-spouse": {
                "name": "marital_status",
                "value": "Married-AF-spouse",
            },
            "marital_status_ Married-civ-spouse": {
                "name": "marital_status",
                "value": "Married-civ-spouse",
            },
            "marital_status_ Married-spouse-absent": {
                "name": "marital_status",
                "value": "Married-spouse-absent",
            },
            "marital_status_ Never-married": {
                "name": "marital_status",
                "value": "Never-married",
            },
            "marital_status_ Separated": {
                "name": "marital_status",
                "value": "Separated",
            },
            "marital_status_ Widowed": {
                "name": "marital_status",
                "value": "Widowed",
            },
            "occupation_ ?": {"name": "occupation", "value": "?"},
            "occupation_ Adm-clerical": {
                "name": "occupation",
                "value": "Adm-clerical",
            },
            "occupation_ Armed-Forces": {
                "name": "occupation",
                "value": "Armed-Forces",
            },
            "occupation_ Craft-repair": {
                "name": "occupation",
                "value": "Craft-repair",
            },
            "occupation_ Exec-managerial": {
                "name": "occupation",
                "value": "Exec-managerial",
            },
            "occupation_ Farming-fishing": {
                "name": "occupation",
                "value": "Farming-fishing",
            },
            "occupation_ Handlers-cleaners": {
                "name": "occupation",
                "value": "Handlers-cleaners",
            },
            "occupation_ Machine-op-inspct": {
                "name": "occupation",
                "value": "Machine-op-inspct",
            },
            "occupation_ Other-service": {
                "name": "occupation",
                "value": "Other-service",
            },
            "occupation_ Priv-house-serv": {
                "name": "occupation",
                "value": "Priv-house-serv",
            },
            "occupation_ Prof-specialty": {
                "name": "occupation",
                "value": "Prof-specialty",
            },
            "occupation_ Protective-serv": {
                "name": "occupation",
                "value": "Protective-serv",
            },
            "occupation_ Sales": {"name": "occupation", "value": "Sales"},
            "occupation_ Tech-support": {
                "name": "occupation",
                "value": "Tech-support",
            },
            "occupation_ Transport-moving": {
                "name": "occupation",
                "value": "Transport-moving",
            },
            "race_ Amer-Indian-Eskimo": {
                "name": "race",
                "value": "Amer-Indian-Eskimo",
            },
            "race_ Asian-Pac-Islander": {
                "name": "race",
                "value": "Asian-Pac-Islander",
            },
            "race_ Black": {"name": "race", "value": "Black"},
            "race_ Other": {"name": "race", "value": "Other"},
            "race_ White": {"name": "race", "value": "White"},
            "sex_ Female": {"name": "sex", "value": "Female"},
            "sex_ Male": {"name": "sex", "value": "Male"},
            "hours_per_week": "hours_per_week",
        },  # get_lending_human_dictionary(),
        "human_data_dict": {
            "income": "income",
            "age": "age",
            "work_class": "work_class",
            "education_years": "education_years",
            "marital_status": "marital_status",
            "occupation": "occupation",
            "race": "race",
            "sex": "sex",
            "hours_per_week": "hours_per_week",
        },  # get_lending_data_dictionary(),
        "X": df_X.values,  # np array #removed to shrink size of dataset obj
        "y": target,  # np array #removed to shrink size of dataset obj
        "use_dummies": use_dummies,
    }

    return dataset


def read_compas_dataset():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    target_label = "class"
    package = files("counterfactual_explainers.data.raw_data")
    csv_path = package / "compas-scores-two-years.csv"
    with csv_path.open("rb") as file:
        df = pd.read_csv(file, skipinitialspace=True)

    columns = [
        "age",
        "sex",
        "race",
        "priors_count",
        "days_b_screening_arrest",
        "c_jail_in",
        "c_jail_out",
        "c_charge_degree",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
        "decile_score",
        "score_text",
    ]

    df = df[columns]
    df["days_b_screening_arrest"] = np.abs(df["days_b_screening_arrest"])
    df["c_jail_out"] = pd.to_datetime(df["c_jail_out"])
    df["c_jail_in"] = pd.to_datetime(df["c_jail_in"])
    df["length_of_stay"] = (df["c_jail_out"] - df["c_jail_in"]).dt.days
    df["length_of_stay"] = np.abs(df["length_of_stay"])
    df["length_of_stay"].fillna(
        df["length_of_stay"].value_counts().index[0], inplace=True
    )
    df["days_b_screening_arrest"].fillna(
        df["days_b_screening_arrest"].value_counts().index[0], inplace=True
    )
    df["length_of_stay"] = df["length_of_stay"].astype(int)
    df["days_b_screening_arrest"] = df["days_b_screening_arrest"].astype(int)
    df["class"] = df["score_text"]
    df.drop(
        ["c_jail_in", "c_jail_out", "decile_score", "score_text"],
        axis=1,
        inplace=True,
    )

    target = df[target_label]
    features = df.drop(columns=[target_label])
    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns
    continuous_features = features.select_dtypes(
        exclude=["object", "category"]
    ).columns

    dataset_name = "compas"
    dataset_config = config["dataset"][dataset_name]
    target_label = dataset_config.get("target_name")
    class_name = target_label
    non_act_features = dataset_config.get("non_act_cols")

    target = df[target_label]
    features = df.drop(columns=[target_label])
    X_columns = features.columns.tolist()
    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns
    continuous_features = features.select_dtypes(
        exclude=["object", "category"]
    ).columns

    continuous = continuous_features.tolist()
    non_continuous = categorical_features.tolist()

    preprocessor, target_encoder = create_data_transformer(
        continuous_features, categorical_features, "minmax", "onehot"
    )

    preprocessor.fit(features)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps[
        "imputer"
    ]
    cat_imputer = preprocessor.named_transformers_["categorical"].named_steps[
        "imputer"
    ]

    # HACK: this is dirty but
    # does the job

    new_feat_cont = cont_imputer.fit_transform(features[continuous_features])
    new_feat_cont = pd.DataFrame(
        new_feat_cont, columns=continuous_features, index=features.index
    )
    if dataset_name != "fico":
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

    df_dce = pd.concat([features, target], axis=1)
    df_X = df_dce[X_columns]

    print(f"NaN values: {features.isnull().values.any()}")

    transformed_target = target_encoder.fit_transform(target)
    possible_outcomes = target_encoder.classes_.tolist()
    labels = tuple(possible_outcomes)
    target = pd.DataFrame(
        transformed_target, columns=[target.name], index=target.index
    )

    type_features, features_type = recognize_features_type(df, class_name)
    df = pd.concat([features, target], axis=1)

    invariants = non_act_features
    binners = []
    label_encoders = []
    use_dummies = True

    continuous_transformer = preprocessor.named_transformers_["continuous"]
    transformed_continuous = continuous_transformer.transform(
        features[continuous_features]
    )

    transformed_df = pd.DataFrame(
        transformed_continuous,
        columns=continuous_features,
        index=features.index,
    )

    scaler = continuous_transformer.named_steps["scaler"]

    idx_features = {i: col for i, col in enumerate(X_columns)}
    mads = {}
    scalers = {}
    for column in continuous_features:
        mad = stats.median_abs_deviation(
            transformed_df[column], nan_policy="omit"
        )
        mads[column] = mad
        scalers[column] = scaler
    # TODO: somehow get dummy dict

    transformed_array = preprocessor.transform(features)
    if isspmatrix_csr(transformed_array):
        transformed_array = transformed_array.toarray()

    feature_names = preprocessor.get_feature_names_out()
    df_X = pd.DataFrame(
        transformed_array, columns=feature_names, index=features.index
    )
    df_X.columns = df_X.columns.str.replace(
        r"^(continuous__|categorical__)", "", regex=True
    )

    # ohe = preprocessor.named_transformers_["categorical"].named_steps[
    # "encoder"
    # ]

    feature_names = np.array(
        [
            re.sub(r"^(continuous__|categorical__)", "", name)
            for name in feature_names
        ]
    )
    dummy = {}
    for column in non_continuous:
        mask = np.array([fn.startswith(column) for fn in feature_names])
        dummy[column] = feature_names[mask]

    dataset = {
        "name": dataset_name,
        "df": df,  # removed to shrink size of dataset obj
        "df_dce": df_dce,  # removed to shrink size of dataset obj
        "columns": df.columns,
        "X_columns": X_columns,
        "X_columns_with_dummies": df_X.columns,
        "class_name": class_name,
        "possible_outcomes": possible_outcomes,
        "type_features": type_features,
        "features_type": features_type,
        #'discrete': discrete,
        "continuous": continuous,
        #'categorical': categorical, #added by JF
        "non_continuous": non_continuous,
        "discrete": non_continuous,
        "dummy": dummy,  # added by JF
        "idx_features": idx_features,
        "label_encoder": label_encoders,
        "scaler": scalers,
        "binner": binners,
        "number_of_bins": 10,
        "mads": mads,
        "labels": labels,
        "invariants": invariants,
        "data_human_dict": {
            "income": "income",
            "age": "age",
            "work_class_ ?": {"name": "work_class", "value": "?"},
            "work_class_ Federal-gov": {
                "name": "work_class",
                "value": "Federal-gov",
            },
            "work_class_ Local-gov": {
                "name": "work_class",
                "value": "Local-gov",
            },
            "work_class_ Never-worked": {
                "name": "work_class",
                "value": "Never-worked",
            },
            "work_class_ Private": {"name": "work_class", "value": "Private"},
            "work_class_ Self-emp-inc": {
                "name": "work_class",
                "value": "Self-emp-inc",
            },
            "work_class_ Self-emp-not-inc": {
                "name": "work_class",
                "value": "Self-emp-not-inc",
            },
            "work_class_ State-gov": {
                "name": "work_class",
                "value": "State-gov",
            },
            "work_class_ Without-pay": {
                "name": "work_class",
                "value": "Without-pay",
            },
            "education_years": "education_years",
            "marital_status_ Divorced": {
                "name": "marital_status",
                "value": "Divorced",
            },
            "marital_status_ Married-AF-spouse": {
                "name": "marital_status",
                "value": "Married-AF-spouse",
            },
            "marital_status_ Married-civ-spouse": {
                "name": "marital_status",
                "value": "Married-civ-spouse",
            },
            "marital_status_ Married-spouse-absent": {
                "name": "marital_status",
                "value": "Married-spouse-absent",
            },
            "marital_status_ Never-married": {
                "name": "marital_status",
                "value": "Never-married",
            },
            "marital_status_ Separated": {
                "name": "marital_status",
                "value": "Separated",
            },
            "marital_status_ Widowed": {
                "name": "marital_status",
                "value": "Widowed",
            },
            "occupation_ ?": {"name": "occupation", "value": "?"},
            "occupation_ Adm-clerical": {
                "name": "occupation",
                "value": "Adm-clerical",
            },
            "occupation_ Armed-Forces": {
                "name": "occupation",
                "value": "Armed-Forces",
            },
            "occupation_ Craft-repair": {
                "name": "occupation",
                "value": "Craft-repair",
            },
            "occupation_ Exec-managerial": {
                "name": "occupation",
                "value": "Exec-managerial",
            },
            "occupation_ Farming-fishing": {
                "name": "occupation",
                "value": "Farming-fishing",
            },
            "occupation_ Handlers-cleaners": {
                "name": "occupation",
                "value": "Handlers-cleaners",
            },
            "occupation_ Machine-op-inspct": {
                "name": "occupation",
                "value": "Machine-op-inspct",
            },
            "occupation_ Other-service": {
                "name": "occupation",
                "value": "Other-service",
            },
            "occupation_ Priv-house-serv": {
                "name": "occupation",
                "value": "Priv-house-serv",
            },
            "occupation_ Prof-specialty": {
                "name": "occupation",
                "value": "Prof-specialty",
            },
            "occupation_ Protective-serv": {
                "name": "occupation",
                "value": "Protective-serv",
            },
            "occupation_ Sales": {"name": "occupation", "value": "Sales"},
            "occupation_ Tech-support": {
                "name": "occupation",
                "value": "Tech-support",
            },
            "occupation_ Transport-moving": {
                "name": "occupation",
                "value": "Transport-moving",
            },
            "race_ Amer-Indian-Eskimo": {
                "name": "race",
                "value": "Amer-Indian-Eskimo",
            },
            "race_ Asian-Pac-Islander": {
                "name": "race",
                "value": "Asian-Pac-Islander",
            },
            "race_ Black": {"name": "race", "value": "Black"},
            "race_ Other": {"name": "race", "value": "Other"},
            "race_ White": {"name": "race", "value": "White"},
            "sex_ Female": {"name": "sex", "value": "Female"},
            "sex_ Male": {"name": "sex", "value": "Male"},
            "hours_per_week": "hours_per_week",
        },  # get_lending_human_dictionary(),
        "human_data_dict": {
            "income": "income",
            "age": "age",
            "work_class": "work_class",
            "education_years": "education_years",
            "marital_status": "marital_status",
            "occupation": "occupation",
            "race": "race",
            "sex": "sex",
            "hours_per_week": "hours_per_week",
        },  # get_lending_data_dictionary(),
        "X": df_X.values,  # np array #removed to shrink size of dataset obj
        "y": target,  # np array #removed to shrink size of dataset obj
        "use_dummies": use_dummies,
    }

    return dataset


# def process_adult():
#     path = "counterfactual_explainers/data/raw_data/adult.csv"
#     df = pd.read_csv(path, skipinitialspace=True, na_values="?")
#
#     X_columns = [
#         "workclass",
#         "age",
#         "marital-status",
#         "occupation",
#         "race",
#         "sex",
#         "hours-per-week",
#     ]
#     y_column = "class"
#     class_name = y_column
#
#     # TODO: do not drop these columns use Imputer instead
#
#     non_continuous = df.select_dtypes(
#         include=["object", "category"]
#     ).columns.tolist()
#
#     non_continuous.remove(class_name)
#
#     print(non_continuous)
#
#     continuous = df.select_dtypes(
#         exclude=["object", "category"]
#     ).columns.tolist()
#
#     print(continuous)
#
#     df = df.dropna(axis=0)
#     df = df.reset_index()
#
#     df = df.drop(
#         labels=[
#             "index",
#             "fnlwgt",
#             "education",
#             "relationship",
#             "capital-gain",
#             "capital-loss",
#             "native-country",
#         ],
#         axis=1,
#     )
#
#     df_X = df[X_columns]
#
#     def get_df_dce(df_in):
#         df_dce = df_in.copy(deep=True)
#         return df_dce
#
#     df_dce = get_df_dce(df)
#
#     # need a unenencoded/scaled dataset for dcf and dce moved below loan status to int
#     # df_dce still has arr_state and years_of credit, if used again will need removing
#
#     invariants = non_act_cols
#     print(type(non_act_cols))
#     binners = []
#     label_encoders = []
#     use_dummies = True
#
#     # TODO: use LabelEncoder
#
#     # gives labels in wrong order with fully paid as 0 and and and charged off as 1 hard code instead
#     df, possible_outcomes, labels = get_possible_outcomes(df, class_name)
#     # possible_outcomes = ['<=50K','50K']#must be in order [0,1]
#     # labels = ('<=50K','50K')#must be in order [0,1]
#     type_features, features_type = recognize_features_type(df, class_name)
#
#     idx_features = {i: col for i, col in enumerate(X_columns)}
#     dummy = {}
#     scalers = {}
#     mads = (
#         {}
#     )  # for use with lime as a fiddle factor to allow lime coeffs to compare continuous to non_continuous , add dict of continuous variables {variable : mad}
#     target = df[class_name]
#
#     # TODO: encode using column transformer
#     for column in df_X.columns:
#
#         if column not in non_continuous:  # for continuous columns
#             # scaling when using dice which requires normalization for compatability
#             print(column)
#             scaler = MinMaxScaler()
#             values = df_X[column].values.reshape(-1, 1)
#             mads[column] = stats.median_abs_deviation(values, axis=None)
#             scaler.fit(values)
#             df_X[column] = scaler.transform(values)
#             scalers[column] = scaler
#
#         else:  # for categorical columns
#             # use get dummies
#             dummy_columns = pd.get_dummies(
#                 df_X[column], prefix=column, prefix_sep="_", drop_first=False
#             ).astype(float)
#             for col in dummy_columns.columns:
#                 df_X[col] = dummy_columns[col]
#             update_dict = {column: dummy_columns.columns.values}
#             dummy.update(update_dict)
#             # encoder = LabelEncoder()
#             # encoder = OneHotEncoder()
#             # encoder.fit(df_X[column].values.reshape(-1,1))
#             df_X.drop(column, axis=1, inplace=True)
#
#     print("VIP info")
#     print(dummy)
#
#     dataset = {
#         "name": file_name.replace(".csv", ""),
#         "df": df,  # removed to shrink size of dataset obj
#         "df_dce": df_dce,  # removed to shrink size of dataset obj
#         "columns": df.columns,
#         "X_columns": X_columns,
#         "X_columns_with_dummies": df_X.columns,
#         "class_name": class_name,
#         "possible_outcomes": possible_outcomes,
#         "type_features": type_features,
#         "features_type": features_type,
#         #'discrete': discrete,
#         "continuous": continuous,
#         #'categorical': categorical, #added by JF
#         "non_continuous": non_continuous,
#         "discrete": non_continuous,
#         "dummy": dummy,  # added by JF
#         "idx_features": idx_features,
#         "label_encoder": label_encoders,
#         "scaler": scalers,
#         "binner": binners,
#         "number_of_bins": 10,
#         "mads": mads,
#         "labels": labels,
#         "invariants": invariants,
#         "data_human_dict": {
#             "income": "income",
#             "age": "age",
#             "work_class_ ?": {"name": "work_class", "value": "?"},
#             "work_class_ Federal-gov": {
#                 "name": "work_class",
#                 "value": "Federal-gov",
#             },
#             "work_class_ Local-gov": {
#                 "name": "work_class",
#                 "value": "Local-gov",
#             },
#             "work_class_ Never-worked": {
#                 "name": "work_class",
#                 "value": "Never-worked",
#             },
#             "work_class_ Private": {"name": "work_class", "value": "Private"},
#             "work_class_ Self-emp-inc": {
#                 "name": "work_class",
#                 "value": "Self-emp-inc",
#             },
#             "work_class_ Self-emp-not-inc": {
#                 "name": "work_class",
#                 "value": "Self-emp-not-inc",
#             },
#             "work_class_ State-gov": {
#                 "name": "work_class",
#                 "value": "State-gov",
#             },
#             "work_class_ Without-pay": {
#                 "name": "work_class",
#                 "value": "Without-pay",
#             },
#             "education_years": "education_years",
#             "marital_status_ Divorced": {
#                 "name": "marital_status",
#                 "value": "Divorced",
#             },
#             "marital_status_ Married-AF-spouse": {
#                 "name": "marital_status",
#                 "value": "Married-AF-spouse",
#             },
#             "marital_status_ Married-civ-spouse": {
#                 "name": "marital_status",
#                 "value": "Married-civ-spouse",
#             },
#             "marital_status_ Married-spouse-absent": {
#                 "name": "marital_status",
#                 "value": "Married-spouse-absent",
#             },
#             "marital_status_ Never-married": {
#                 "name": "marital_status",
#                 "value": "Never-married",
#             },
#             "marital_status_ Separated": {
#                 "name": "marital_status",
#                 "value": "Separated",
#             },
#             "marital_status_ Widowed": {
#                 "name": "marital_status",
#                 "value": "Widowed",
#             },
#             "occupation_ ?": {"name": "occupation", "value": "?"},
#             "occupation_ Adm-clerical": {
#                 "name": "occupation",
#                 "value": "Adm-clerical",
#             },
#             "occupation_ Armed-Forces": {
#                 "name": "occupation",
#                 "value": "Armed-Forces",
#             },
#             "occupation_ Craft-repair": {
#                 "name": "occupation",
#                 "value": "Craft-repair",
#             },
#             "occupation_ Exec-managerial": {
#                 "name": "occupation",
#                 "value": "Exec-managerial",
#             },
#             "occupation_ Farming-fishing": {
#                 "name": "occupation",
#                 "value": "Farming-fishing",
#             },
#             "occupation_ Handlers-cleaners": {
#                 "name": "occupation",
#                 "value": "Handlers-cleaners",
#             },
#             "occupation_ Machine-op-inspct": {
#                 "name": "occupation",
#                 "value": "Machine-op-inspct",
#             },
#             "occupation_ Other-service": {
#                 "name": "occupation",
#                 "value": "Other-service",
#             },
#             "occupation_ Priv-house-serv": {
#                 "name": "occupation",
#                 "value": "Priv-house-serv",
#             },
#             "occupation_ Prof-specialty": {
#                 "name": "occupation",
#                 "value": "Prof-specialty",
#             },
#             "occupation_ Protective-serv": {
#                 "name": "occupation",
#                 "value": "Protective-serv",
#             },
#             "occupation_ Sales": {"name": "occupation", "value": "Sales"},
#             "occupation_ Tech-support": {
#                 "name": "occupation",
#                 "value": "Tech-support",
#             },
#             "occupation_ Transport-moving": {
#                 "name": "occupation",
#                 "value": "Transport-moving",
#             },
#             "race_ Amer-Indian-Eskimo": {
#                 "name": "race",
#                 "value": "Amer-Indian-Eskimo",
#             },
#             "race_ Asian-Pac-Islander": {
#                 "name": "race",
#                 "value": "Asian-Pac-Islander",
#             },
#             "race_ Black": {"name": "race", "value": "Black"},
#             "race_ Other": {"name": "race", "value": "Other"},
#             "race_ White": {"name": "race", "value": "White"},
#             "sex_ Female": {"name": "sex", "value": "Female"},
#             "sex_ Male": {"name": "sex", "value": "Male"},
#             "hours_per_week": "hours_per_week",
#         },  # get_lending_human_dictionary(),
#         "human_data_dict": {
#             "income": "income",
#             "age": "age",
#             "work_class": "work_class",
#             "education_years": "education_years",
#             "marital_status": "marital_status",
#             "occupation": "occupation",
#             "race": "race",
#             "sex": "sex",
#             "hours_per_week": "hours_per_week",
#         },  # get_lending_data_dictionary(),
#         "X": df_X.values,  # np array #removed to shrink size of dataset obj
#         "y": target,  # np array #removed to shrink size of dataset obj
#         "use_dummies": use_dummies,
#     }
#
#     return dataset
#


def recognize_features_type(df, class_name):
    integer_features = list(df.select_dtypes(include=["int64"]).columns)
    double_features = list(df.select_dtypes(include=["float64"]).columns)
    string_features = list(df.select_dtypes(include=["object"]).columns)
    type_features = {
        "integer": integer_features,
        "double": double_features,
        "string": string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = "integer"
    for col in double_features:
        features_type[col] = "double"
    for col in string_features:
        features_type[col] = "string"

    return type_features, features_type


# def process_compas():
#     path = "datasets/"
#     file_name = "compas.csv"  #'vate '
#     url = path + file_name
#     df = pd.read_csv(url)
#
#     selected_columns = [
#         "sex",
#         "age",
#         "race",
#         "c_charge_degree",
#         "priors_count",
#         "score_text",
#     ]
#     X_columns = ["sex", "age", "race", "c_charge_degree", "priors_count"]
#     y_column = (
#         "score_text"  # needs transforming to 0 for not High and 1 for High
#     )
#
#     class_name = y_column
#     non_continuous = ["sex", "race", "c_charge_degree"]
#     continuous = ["age", "priors_count"]
#
#     df = df.drop(
#         labels=[
#             "id",
#             "name",
#             "first",
#             "last",
#             "compas_screening_date",
#             "dob",
#             "age_cat",
#             "juv_fel_count",
#             "juv_misd_count",
#             "decile_score",
#             "juv_other_count",
#             "days_b_screening_arrest",
#             "c_jail_in",
#             "c_jail_out",
#             "c_case_number",
#             "c_offense_date",
#             "c_arrest_date",
#             "c_days_from_compas",
#             "c_charge_desc",
#             "is_recid",
#             "num_r_cases",
#             "r_case_number",
#             "r_charge_degree",
#             "r_days_from_arrest",
#             "r_offense_date",
#             "r_charge_desc",
#             "r_jail_in",
#             "r_jail_out",
#             "is_violent_recid",
#             "num_vr_cases",
#             "vr_case_number",
#             "vr_charge_degree",
#             "vr_offense_date",
#             "vr_charge_desc",
#             "v_type_of_assessment",
#             "v_decile_score",
#             "v_score_text",
#             "v_screening_date",
#             "type_of_assessment",
#             "decile_score.1",
#             "screening_date",
#         ],
#         axis=1,
#     )
#     df = df.dropna(axis=0)
#     df = df.reset_index()
#     df_X = df[X_columns]
#     target = df[class_name]
#     for i in range(len(df[class_name])):
#         if df[class_name][i] == ("High"):
#             df[class_name][i] = "Recidivist"
#             # target[i] = 1
#         else:
#             df[class_name][i] = "Non-recidivist"
#             # target[i] = 0
#
#     def get_df_dce(df_in):
#         df_dce = df_in.copy(deep=True)
#         # df_dce.drop(columns = 'index',inplace=True)
#         return df_dce
#
#     df_dce = get_df_dce(
#         df
#     )  # need a unenencoded/scaled dataset for dcf and dce moved below loan status to int
#     # df_dce still has arr_state and years_of credit, if used again will need removing
#     invariants = []
#     binners = []
#     label_encoders = []
#     use_dummies = True
#     # gives labels in wrong order with fully paid as 0 and and and charged off as 1 hard code instead
#     df, possible_outcomes, labels = get_possible_outcomes(df, class_name)
#     # possible_outcomes = ['<=50K','50K']#must be in order [0,1]
#     # labels = ('<=50K','50K')#must be in order [0,1]
#     type_features, features_type = prepare_dataset.recognize_features_type(
#         df, class_name
#     )
#     idx_features = {i: col for i, col in enumerate(X_columns)}
#     dummy = {}
#     scalers = {}
#     mads = (
#         {}
#     )  # for use with lime as a fiddle factor to allow lime coeffs to compare continuous to non_continuous , add dict of continuous variables {variable : mad}
#
#     for column in df_X.columns:
#         if column not in non_continuous:  # for continuous columns
#             # scaling when using dice which requires normalization for compatability
#             scaler = MinMaxScaler()
#             values = df_X[column].values.reshape(-1, 1)
#             mads[column] = stats.median_absolute_deviation(values, axis=None)
#             scaler.fit(values)
#             df_X[column] = scaler.transform(values)
#             scalers[column] = scaler
#
#             """
#             #change to standardising continuous columns, use this when not using dice
#             #scaler = StandardScaler()#used at start of big_survey
#             scaler = StandardScaler()#defaults to range (0,1) this is sclaer for trying on test of big survey
#             values = (df_X[column].values.reshape(-1,1))
#             mads[column] = stats.median_absolute_deviation(values, axis = None)
#             scaler.fit(values)
#             df_X[column] = scaler.transform(values)
#             scalers[column] = scaler
#             """
#
#         else:  # for categorical columns
#             # use get dummies
#             dummy_columns = pd.get_dummies(
#                 df_X[column], prefix=column, prefix_sep="_", drop_first=False
#             )
#             for col in dummy_columns.columns:
#                 df_X[col] = dummy_columns[col]
#             update_dict = {column: dummy_columns.columns.values}
#             dummy.update(update_dict)
#             # encoder = LabelEncoder()
#             # encoder = OneHotEncoder()
#             # encoder.fit(df_X[column].values.reshape(-1,1))
#             df_X.drop(column, axis=1, inplace=True)
#     dataset = {
#         "name": file_name.replace(".csv", ""),
#         "df": df,  # removed to shrink size of dataset obj
#         "df_dce": df_dce,  # removed to shrink size of dataset obj
#         "columns": df.columns,
#         "X_columns": X_columns,
#         "X_columns_with_dummies": df_X.columns,
#         "class_name": class_name,
#         "possible_outcomes": possible_outcomes,
#         "type_features": type_features,
#         "features_type": features_type,
#         #'discrete': discrete,
#         "continuous": continuous,
#         #'categorical': categorical, #added by JF
#         "non_continuous": non_continuous,
#         "discrete": non_continuous,
#         "dummy": dummy,  # added by JF
#         "idx_features": idx_features,
#         "label_encoder": label_encoders,
#         "scaler": scalers,
#         "binner": binners,
#         "number_of_bins": 10,
#         "mads": mads,
#         "labels": labels,
#         "invariants": invariants,
#         "data_human_dict": {
#             "age": "age",
#             "race_African-American": {
#                 "name": "race",
#                 "value": "African-American",
#             },
#             "race_Asian": {"name": "race", "value": "Asian"},
#             "race_Caucasian": {"name": "race", "value": "Caucasian"},
#             "race_Hispanic": {"name": "race", "value": "Hispanic"},
#             "race_Native American": {
#                 "name": "race",
#                 "value": "Native American",
#             },
#             "race_Other": {"name": "race", "value": "Native American"},
#             "priors_count": "priors_count",
#             "sex_Female": {"name": "sex", "value": "Female"},
#             "sex_Male": {"name": "sex", "value": "Male"},
#             "c_charge_degree_F": {"name": "c_charge_degree", "value": "F"},
#             "c_charge_degree_M": {"name": "c_charge_degree", "value": "M"},
#             "c_charge_degree_O": {"name": "c_charge_degree", "value": "O"},
#         },
#         "human_data_dict": {
#             "age": "age",
#             "race": "race",
#             "priors_count": "priors_count",
#             "sex": "sex",
#             "c_charge_degree": "c_charge_degree",
#         },
#         "X": df_X.values,  # np array #removed to shrink size of dataset obj
#         "y": target,  # np array #removed to shrink size of dataset obj
#         "use_dummies": use_dummies,
#     }
#     pickle_filename = "pickled_data/compas_pickled_data_MinMax_6_07_22.p"
#     outfile = open(pickle_filename, "wb")
#     pickle.dump(dataset, outfile)
#     outfile.close()
#
#     return dataset
