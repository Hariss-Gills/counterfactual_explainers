import re
import sqlite3

import numpy as np


def create_connection(db_file):
    """create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    from sqlite3 import Error

    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None


def descale_db(k, val_string, dataset):
    scaler = dataset["scaler"][k]
    val_float = float(val_string)
    """
    val_float = scaler.inverse_transform([val_float])
    val = str(val_float[0])
    """
    val_float = scaler.inverse_transform([[val_float]])
    val = str(val_float[0][0])
    return val


def decode_db(k, val_string, dataset):
    coder = dataset["label_encoder"][k]
    new_val = coder.inverse_transform(val_string)
    return new_val


def debin_db(k, val_string, dataset):
    binner = dataset["binner"]
    v = np.array(float(val_string))
    v = binner[k].inverse_transform(v.reshape(1, -1))  # get center of bucket

    v = v[0][0]
    for i in range(len(binner[k].bin_edges_[0])):
        if (
            binner[k].bin_edges_[0][i] <= v
        ):  # what if in last bucket? how to get upper bound?
            count = i
        else:
            break
    # val_string= str(binner[k].bin_edges_[0][count])
    if count < dataset["number_of_bins"]:
        # change so that when bin edges are 1 apert we donot get between 2.0 and 2.0 but 2.0
        if (
            binner[k].bin_edges_[0][count]
            == binner[k].bin_edges_[0][count + 1] - 1
        ):
            val_string = str(binner[k].bin_edges_[0][count])
        else:
            val_string = (
                "between "
                + str(binner[k].bin_edges_[0][count])
                + " and "
                + str((binner[k].bin_edges_[0][count + 1]) - 1)
            )
    else:
        val_string = "greater than or equal to " + str(
            binner[k].bin_edges_[0][count]
        )
    return val_string


# new function bin_db() here
# in: value in mixed polytope -8 or -7 or +ve numerical
# out binned value or string for special value
def bin_db(k, val_string, dataset):
    # no decoding required or wanted
    # value assumed to be an string make a float
    value = val_string
    # print('k: ', k,' val_string: ',val_string)

    if k in dataset["binner"]:
        value = int(val_string)
        if value >= 0:  # all special values <0
            # value = debin_db(k,val_arr,dataset)no wanted or neccessary
            # bin values
            value = dataset["binner"][k].transform(
                [[value]]
            )  # value mustr be 2D array
            # then debin to keep data consistent
            value = debin_db(k, value[0], dataset)  # needs to be a 1D array
            # what if change is not enough to change bin? eg bin is 1-3 and delta is from 1 to 2?
        else:
            value = special_values(value)
    else:  # is_categorical
        if k == "MaxDelq2PublicRecLast12M":
            value = int(val_string)
            value = MaxDelq2PublicRecLast12M(value)
        if k == "MaxDelqEver":
            value = int(val_string)
            value = MaxDelqEver(value)
    return value


def special_values(value):
    # using definition of special values from https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=413df
    if value == -9:
        value = "No Bureau Record or No Investigation"
    elif value == -8:
        value = "No usable values"
    elif value == -7:
        value = "Condition not met (e.g. no inquiries, no late payments)"
    else:
        value = str(value)
    return value


def MaxDelq2PublicRecLast12M(value):
    # special method to return 'MaxDelq2PublicRecLast12M' categorical values for fico HELOC dataset, not generalisable
    if value == 0:
        # value = 'derogatory comment'
        value = "derogatory comment(s) in public records"
    elif value == 1:
        # value = '120+ days delinquent'
        value = "payment of 120+ days late in public record"
    elif value == 2:
        # value =  '90 days delinquent'
        value = "payment of 90 days late in public record"
    elif value == 3:
        # value =  '60 days delinquent'
        value = "payment of 60 days late in public record"
    elif value == 4:
        # value =  '30 days delinquent'
        value = "payment of 30 days late in public record"
    elif value == 5 or 6:
        # value =  'unknown delinquency'
        value = "no known late payments in public record"
    elif value == 7:
        value = "current and never a late payment in public "
        # value =  'current and never delinquent'
    elif value == 8 or 9:
        value = "all other"
    else:
        value = str(value)
    return value


def MaxDelqEver(value):
    # special method to return 'MaxDelqEver' categorical values for fico HELOC dataset, not generalisable

    if value == 1:
        value = "No such value"
    elif value == 2:
        value = "derogatory comment"
    elif value == 3:
        # value =  '120+ days delinquent'
        value = "120+ days late payment"
    elif value == 4:
        # value =  '90 days delinquent'
        value = "90 days late payment"
    elif value == 5:
        # value =  '60 days delinquent'
        value = "60 days late payment"
    elif value == 6:
        # value =  '30 days delinquent'
        value = "30 days late payment"
    elif value == 7:
        # value =  'unknown delinquency'
        value = "no known late payments"
    elif value == 8:
        # value =  'current and never delinquent'
        value = "current and never paid late"
    elif value == 9:
        value = "all other"
    else:
        value = str(value)
    return value


def decode_debin_db(k, val_string, dataset, decode=True):
    if decode:
        val_arr = [val_string]  # decode needs an array
        if k in dataset["label_encoder"]:
            val_arr = decode_db(k, val_arr, dataset)[0]
        val_arr = val_arr[0]
    else:
        val_arr = val_string
    """
    else:
        value = val_arr[0]#transform back to value from array. change name? no keep return value simple
    """
    if k in dataset["binner"]:
        if val_arr >= 0:  # all special values <0
            val_arr = debin_db(k, val_arr, dataset)
        else:
            # using definition of special values from https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=413df
            if val_arr == -9:
                val_arr = "No Bureau Record or No Investigation"
            elif val_arr == -8:
                val_arr = "No usable values"
            elif val_arr == -7:
                val_arr = (
                    "Condition not met (e.g. no inquiries, no late payments)"
                )
            else:
                val_arr = str(val_arr)
        # else:#will be is_categorical
        if k == "MaxDelq2PublicRecLast12M":
            if val_arr == 0:
                # val_arr = 'derogatory comment'
                val_arr = "derogatory comment(s) in public records"
            elif val_arr == 1:
                # val_arr = '120+ days delinquent'
                val_arr = "payment of 120+ days late in public record"
            elif val_arr == 2:
                # val_arr =  '90 days delinquent'
                val_arr = "payment of 90 days late in public record"
            elif val_arr == 3:
                # val_arr =  '60 days delinquent'
                val_arr = "payment of 60 days late in public record"
            elif val_arr == 4:
                # val_arr =  '30 days delinquent'
                val_arr = "payment of 30 days late in public record"
            elif val_arr == 5 or 6:
                # val_arr =  'unknown delinquency'
                val_arr = "no known late payments in public record"
            elif val_arr == 7:
                val_arr = "current and never a late payment in public "
                # val_arr =  'current and never delinquent'
            elif val_arr == 8 or 9:
                val_arr = "all other"
            else:
                val_arr = str(val_arr)
        if k == "MaxDelqEver":
            if val_arr == 1:
                val_arr = "No such value"
            elif val_arr == 2:
                val_arr = "derogatory comment"
            elif val_arr == 3:
                # val_arr =  '120+ days delinquent'
                val_arr = "120+ days late payment"
            elif val_arr == 4:
                # val_arr =  '90 days delinquent'
                val_arr = "90 days late payment"
            elif val_arr == 5:
                # val_arr =  '60 days delinquent'
                val_arr = "60 days late payment"
            elif val_arr == 6:
                # val_arr =  '30 days delinquent'
                val_arr = "30 days late payment"
            elif val_arr == 7:
                # val_arr =  'unknown delinquency'
                val_arr = "no known late payments"
            elif val_arr == 8:
                # val_arr =  'current and never delinquent'
                val_arr = "current and never paid late"
            elif val_arr == 9:
                val_arr = "all other"
            else:
                val_arr = str(val_arr)
    if k in dataset["scaler"]:
        val_arr = descale_db(
            k, val_arr, dataset
        )  # val_arr is a value not an array
    return val_arr


def db_add_explanations(conn, description, record_number):
    description = str(description)
    record_number = int(record_number)
    sql = """INSERT INTO explanations (description,record_number)
            VALUES(?,?)  """
    cur = conn.cursor()
    cur.execute(sql, (description, record_number))
    return cur.lastrowid


def db_add_records(conn, explanations_id):
    sql = """INSERT INTO records (explanation_id)
            VALUES(?)   """
    cur = conn.cursor()
    cur.execute(sql, (explanations_id,))
    return cur.lastrowid


def db_add_attribute_types(conn, attribute_type):
    sql = """INSERT INTO attribute_types (attribute_name)
            VALUES(?)   """
    cur = conn.cursor()
    cur.execute(sql, (attribute_type,))
    return cur.lastrowid


def db_add_operands(conn, operand):
    sql = """INSERT INTO operands (operand_value)
            VALUES(?)   """
    cur = conn.cursor()
    cur.execute(sql, (operand,))
    return cur.lastrowid


def db_add_attributes_operands_types(conn, attribute_type_id, operand_id):
    sql = """INSERT INTO attributes_operands_types (attribute_type_id,operand_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            attribute_type_id,
            operand_id,
        ),
    )
    return cur.lastrowid


def db_add_attributes(conn, attributes_operands_types_id, attribute_value):
    sql = """INSERT INTO attributes (attributes_operands_types_id,attribute_value)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            attributes_operands_types_id,
            attribute_value,
        ),
    )
    return cur.lastrowid


def db_add_records_attributes(conn, attributes_id, records_id):
    sql = """INSERT INTO records_attributes(attribute_id,record_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            attributes_id,
            records_id,
        ),
    )
    return cur.lastrowid


def db_add_invariants_attributes(conn, attributes_id, records_id):
    sql = """INSERT INTO invariants_attributes(attribute_id,record_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            attributes_id,
            records_id,
        ),
    )
    return cur.lastrowid


def db_add_rules(conn, explanations_id):
    sql = """INSERT INTO rules (explanation_id)
            VALUES(?)  """
    cur = conn.cursor()
    cur.execute(sql, (explanations_id,))
    return cur.lastrowid


def db_add_rules_attributes(conn, rules_id, attributes_id):
    sql = """INSERT INTO rules_attributes(rule_id,attribute_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            rules_id,
            attributes_id,
        ),
    )
    return cur.lastrowid


def db_add_deltas(conn, explanations_id):
    sql = """INSERT INTO deltas (explanation_id)
            VALUES(?)  """
    cur = conn.cursor()
    cur.execute(sql, (explanations_id,))
    return cur.lastrowid


def db_add_deltas_attributes(conn, deltas_id, attributes_id):
    sql = """INSERT INTO attributes_deltas(delta_id,attribute_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            deltas_id,
            attributes_id,
        ),
    )
    return cur.lastrowid


def db_add_differences_attributes(conn, deltas_id, attributes_id):
    sql = """INSERT INTO differences_attributes(delta_id,attribute_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            deltas_id,
            attributes_id,
        ),
    )
    return cur.lastrowid


def db_add_importances(conn, explanations_id):
    sql = """INSERT INTO importances (explanation_id)
            VALUES(?)  """
    cur = conn.cursor()
    cur.execute(sql, (explanations_id,))
    return cur.lastrowid


def db_add_importances_attributes(conn, importances_id, attributes_id):
    sql = """INSERT INTO importances_attributes(importance_id,attribute_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            importances_id,
            attributes_id,
        ),
    )
    return cur.lastrowid


def db_add_class_probabilities(conn, explanations_id):
    sql = """INSERT INTO class_probabilities (explanation_id)
            VALUES(?)  """
    cur = conn.cursor()
    cur.execute(sql, (explanations_id,))
    return cur.lastrowid


def db_add_class_probabilities_attributes(
    conn, class_probability_id, attributes_id
):
    sql = """INSERT INTO class_probabilities_attributes(class_probability_id,attribute_id)
            VALUES(?,?)   """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            class_probability_id,
            attributes_id,
        ),
    )
    return cur.lastrowid


def db_add_attributes_bundle(conn, att_type, operand, att_val):
    # change att_type to human friendly value

    fico_human_dict = {
        "RiskPerformance": "Decision on awarding your application",
        "ExternalRiskEstimate": "External estimated score for your application",
        "MSinceOldestTradeOpen": "Age of your oldest credit in months",
        "MSinceMostRecentTradeOpen": "Age of your newest credit in months",
        "AverageMInFile": "Average age of your credits in months",
        "NumSatisfactoryTrades": "Number of credits you have repaid in full and on time",
        "NumTrades60Ever2DerogPubRec": "Number of credits you have repaid 60 days late",
        "NumTrades90Ever2DerogPubRec": "Number of credits you have repaid 90 days late",
        "PercentTradesNeverDelq": "Percentage of your credits with no late payments",
        "MSinceMostRecentDelq": "Months since your most recent late payment",
        "MaxDelq2PublicRecLast12M": "Latest payment on your credits in public records in the last 12 Months",
        "MaxDelqEver": "Latest payment on your credits",
        "NumTotalTrades": "Your total number of credits ever",
        "NumTradesOpeninLast12M": "Number of credits taken by you in last 12 months",
        "PercentInstallTrades": "Percentage of your credits paid by installment",
        "MSinceMostRecentInqexcl7days": "Months since last inquiry (excluding last 7 days) about credit for you",
        "NumInqLast6M": "Total number of inquiries in last 6 months about credit for you",
        "NumInqLast6Mexcl7days": "Total number of inquiries in last 6 months (excluding last 7 days) about credit for you",
        "NetFractionRevolvingBurden": "Balance of your credit cards, divided by your credit limit, as a percentage",
        "NetFractionInstallBurden": "Your installment balance divided by your original credit amount as a percentage",
        "NumRevolvingTradesWBalance": "Your number of credit cards with a balance still to pay",
        "NumInstallTradesWBalance": "Your number of installment credits with a balance still to pay",
        "NumBank2NatlTradesWHighUtilization": "The number of your credits near to their limit",
        "PercentTradesWBalance": "Percentage of your credits with a balance to pay",
    }
    if att_type in fico_human_dict.keys():
        att_type = fico_human_dict[att_type]
    attributes_type_id = db_add_attribute_types(
        conn, att_type
    )  # write human desc to db
    operands_id = db_add_operands(conn, operand)
    attributes_operands_types_id = db_add_attributes_operands_types(
        conn, attributes_type_id, operands_id
    )
    attributes_id = db_add_attributes(
        conn, attributes_operands_types_id, att_val
    )
    return attributes_id


# def seperate_single_value:


def seperate_double_value(v, op_chars):
    # import pdb; pdb.set_trace()
    op_string_a = "<"
    op_string_b = "<="
    op_indices = []
    # always start with non op_char
    # always of form value < att_name <= value
    op_indices.append(v.find("<"))
    op_indices.append(v.find("="))
    val_string_a = v[0 : (op_indices[0])]
    val_string_b = v[op_indices[1] + 1 : (len(v))]
    att_name = v[
        (op_indices[0]) : (op_indices[1] - 2)
    ]  # att_name not necessary
    return op_string_a, op_string_b, val_string_a, val_string_b, att_name


def seperate_operand_value(v):
    # loop characters c in string v
    op_string = ""
    val_string = ""
    op_chars = ["<", ">", "="]
    op_array = []
    val_array = []
    # i = 0
    for c in v:
        if c in op_chars:
            op_string = op_string + c
            # i = i+1
        else:
            val_string = val_string + c
    return op_string, val_string
