# translation and adaption of Brownlee Ruby code
# http://www.cleveralgorithms.com/nature-inspired/immune/immune_network_algorithm.html

import copy
import csv
import gc
import math
import sqlite3
from operator import itemgetter

import keras.backend
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from counterfactual_explainers.aide import create_db, predict, write_to_db


def set_parameter_dict():
    # return parameter_dict
    parameter_dict = {
        "sort_by": "distance",
        "use_mads": True,
        "problem_size": 1,
        "search_space": [0, 1],
        "max_gens": 5,
        "pop_size": 20,
        "num_clones": 10,
        "beta": 1,
        "num_rand": 2,
        "affinity_constant": 0.35,
        "stop_condition": 0.01,
        "new_cell_rate": 0.4,
    }
    return parameter_dict


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


def get_line_columns(dataset):
    # used to get columns for use with onehot encoded dummies, idf dummies used else without dummied columns in list.
    # params dataset dict object
    # returns list of columns titles
    line_columns = dataset["X_columns"]
    if dataset["use_dummies"] == True:
        line_columns = dataset["X_columns_with_dummies"]
    return line_columns


def keras_pred(model, line):
    # return the value the model predicts for the inputs
    # param the keras model, the input values
    # return array for [prediction for 0, perdiction for 1]
    # keras 3 new line
    # pred = float(predict.predict_2D(model, line)) old line
    pred = predict.predict_batch(model, line)
    pred = (np.argmax(pred, axis=1) == 0).astype(int)
    # pred = float(predict.predict_batch(model, line))
    pred = float(pred)
    pred_array = [1 - pred, pred]

    return pred_array


def cost_constraint(cost):
    # return False if cost greater than value 0 for binary classifier else True
    # returning False exits a while loop in calling method
    if cost > 0:
        return False
    else:
        return True


def select_random_value(col, dataset):
    # select a random value for a a piece of data in a given column
    # 3 cases depending if a coulm is label_encoded, binned, or one-hot encoded(dummied), or scaled
    # params(column name, dataset dict object)
    # returns a randomly selected value
    ran = np.random.rand()  # ran is a random number between 0 and 1

    if col in dataset["label_encoder"]:
        num_classes = len(dataset["label_encoder"][col].classes_)
        value = math.trunc(ran * num_classes)
    elif col in dataset["binner"]:  # if not in label encoder then binned
        number_of_bins = dataset[
            "number_of_bins"
        ]  # number of bins how to automate this number?
        value = trunc(ran * number_of_bins)
    elif col in dataset["dummy"].keys():
        dummy = dataset["dummy"][col]
        num_dummies = len(dummy)
        dummy_num = math.trunc(ran * num_dummies)
        value = dummy[dummy_num]
    elif col in dataset["scaler"]:  # is continuous variable
        value = round(ran, 1)  # not called for continuous varibles
    else:
        value = round(ran, 1)  # not called for continuous varibles
    return value


def new_cell(parameter_dict, dataset, model, prediction, target_vals):
    # create a new cell
    # params dataset dict object, ml model, predicted result (int), values of input instance
    # return a new cell object (dict)

    # add new constraint a count that when it reaches a limit just returns a null cell
    count = 0
    limit = 100
    flag = True
    while (
        flag
    ):  # while combined distance == 0 flag is true , flag to false  i.e. must be different to reference cell
        count = count + 1
        # print('in new cell loop count = : ',count)
        line = []
        X_columns = dataset["X_columns"]
        invariants = dataset["invariants"]
        continuous = list()
        non_continuous = list()
        for col in X_columns:
            if col in dataset["continuous"]:
                continuous.append(col)
            else:
                non_continuous.append(col)
        columns = list()
        # is the following block necessary?yes first 4 columns are continuous and must be fed to NN first and in order
        for c in continuous:
            columns.append(c)
        for nc in non_continuous:
            columns.append(nc)
        dummy = dataset["dummy"]
        index = 0  # index for dummy columns
        for i in range(len(columns)):
            if columns[i] in continuous:
                if columns[i] in invariants:
                    line = np.append(line, target_vals[index])
                    index = index + 1
                else:  # non invariant e.g. variable
                    value = np.random.rand()
                    # value = round(value,1) no pint rounding here as effect of mutation is to move by small amounts
                    line = np.append(line, value)
                    index = index + 1
            elif columns[i] in non_continuous:
                if columns[i] in dummy.keys():
                    if columns[i] in invariants:
                        for dum in dummy[columns[i]]:
                            line = np.append(line, target_vals[index])
                            index = index + 1
                    else:  # non invariants
                        selected_value = select_random_value(
                            columns[i], dataset
                        )
                        for dum in dummy[columns[i]]:
                            if dum == selected_value:
                                line = np.append(line, 1)
                                index = index + 1
                            else:
                                line = np.append(line, 0)
                                index = index + 1
                else:  # non dummies in this branch
                    if columns[i] in invariants:
                        line = np.append(line, target_vals[index])
                        index = index + 1
                    else:  # non invariants
                        value = select_random_value(columns[i], dataset)
                        line = np.append(line, value)
                        index = index + 1

        line = line.reshape(1, -1)
        cost = objective_function(line, model, prediction)
        flag = cost_constraint(cost)
        # must rule out cells with a distance of 0
        if flag == False:
            if (
                combined_distance(
                    parameter_dict, target_vals, line[0], dataset
                )
                == 0
            ):
                flag = True  # do not allow cells with zero distance as these must have the same values as the  record
                breakpoint()
        if count > limit:
            # print('count > limit for new_cell')
            cell = None
            return cell
    dist_to_target = combined_distance(
        parameter_dict, target_vals, line[0], dataset
    )
    cell = {
        "value": line,
        "norm_cost": 0,
        "cost": cost,
        # 'norm_distance':0,
        "distance": dist_to_target,
    }
    return cell


def create_mutated_cell(
    parameter_dict, value, norm_cost, model, prediction, dataset, target_vals
):  # value is an array
    # take in a cells value and normalised cost and then create a mutation of that cell until it has a lower normalised cost
    # params (value array, norm_cost float, prediction int, dayset dict object, target vals array)
    # return a cell with a lower normalised cost
    cost = objective_function(value, model, prediction)
    dist_to_target = combined_distance(
        parameter_dict, target_vals, value[0], dataset
    )  # 'fault here some distances of clones == 0'
    cell = {
        "value": value,
        "norm_cost": norm_cost,
        "cost": cost,
        "distance": dist_to_target,
    }
    return cell


def objective_function(line, model, prediction):
    # objective function returns a prediction of a line of values from a model
    # params line array, model ml model from keras, prediction int
    # returns  a value beteeen -0.5 to 0.5
    # if pred is 0 to get cf line[0][0] should be < 0.5
    # if pred is 1 to get cf line[0][1] should be < 0.5
    # for scikit
    # result =  ((1 - (model.predict_proba(line)[0][prediction]))-0.5)*2
    # for keras
    # print('objective function to called')

    result = 0.5 - keras_pred(model, line)[prediction]
    keras.backend.clear_session()  # clearing up keras tf models to prevent mem leak
    return result


def random_vector(minmax):  # minmax is a vector of two element array
    rand = (2 * np.random.rand()) - 1
    value = (minmax[0] + (minmax[1] - minmax[0])) * rand
    return value


# replaced  by np.random.normal()
# reinserted because np.random.normal(loc=0.0,scale=1.0,size= 1) does not work
def random_gaussian(mean=0.0, stdev=1.0):
    return np.random.normal()


def clone(parent):  # parent is an array
    clone = parent
    return clone


def mutation_rate(beta, normalized_cost):
    return (1.0 / beta) * np.exp(-normalized_cost)


def mutate(
    parameter_dict, child, normalized_cost, dataset, model, prediction
):  # child is a 2d array [i][v]
    count = 0
    limit = 10  # if count = limit terminate process
    alpha = mutation_rate(parameter_dict["beta"], normalized_cost)
    columns = list()
    for c in dataset["continuous"]:
        columns.append(c)
    for nc in dataset["non_continuous"]:
        columns.append(nc)
    flag = True
    line = copy.deepcopy(child["value"])
    dummy = dataset["dummy"]
    # to accommadate dummies this will be easier to do in a dataframe as column names will be easier to manipulate than column numbers

    line_columns = get_line_columns(dataset)
    df_line = pd.DataFrame(data=line, columns=line_columns)
    while flag:
        # potential for counter to stop long loops if more than given number return original cell
        for i in range(len(columns)):  # -1 because class name not in line
            if columns[i] in dataset["continuous"]:
                if columns[i] not in dataset["invariants"]:
                    # add a constraint for mutate by distance to ensure value of feature is between 0 & 1
                    feature_value = df_line[columns[i]] + (
                        alpha * random_gaussian(mean=0.0, stdev=1.0)
                    )
                    if feature_value[0] < 0:
                        feature_value[0] = 0
                    elif feature_value[0] > 1:
                        feature_value[0] = 1
                    df_line[columns[i]] = feature_value

            if columns[i] in dataset["non_continuous"]:
                if columns[i] not in dataset["invariants"]:
                    r_line = (
                        np.random.rand()
                    )  # picks a value to change to may be the original
                    r_alpha = np.random.rand()  # chance of mutation happening
                    # num_classes = len(dataset['label_encoder'][columns[i]].classes_)
                    # mutate to random value if r < alpha
                    # below is weird alpha is prop to 1/beta why multiply beta out?
                    # print('get non-continuous mutation')
                    if (
                        r_alpha < 10 * parameter_dict["beta"] * alpha
                    ):  # 10*alpha to makea mutation more likely
                        value = select_random_value(columns[i], dataset)
                        if columns[i] in dummy.keys():
                            for dum in dummy[columns[i]]:
                                if dum == value:
                                    df_line[dum] = 1
                                else:

                                    df_line[dum] = 0
                        else:
                            df_line[columns[i]] = value
        flag = cost_constraint(
            objective_function(df_line.values, model, prediction)
        )

        if (
            flag == False
        ):  # check distance if 0 not allowed why is cost constraint allowing this?
            if (
                combined_distance(
                    parameter_dict, child["value"][0], df_line.values, dataset
                )
                == 0
            ):
                flag = True
                print(
                    "error: distance between target cell and mutated cell is 0"
                )
                breakpoint()
        count = count + 1
        if count > limit:
            return child["value"]
    line = df_line.values
    gc.collect()
    return line


def sort_by_dist(list_of_dict):
    sorted_list = sorted(list_of_dict, key=itemgetter("distance", "cost"))
    # sort by cost added to differentiate between categorical rich data which tends to have similar distances
    return sorted_list


def sort_by_cost(list_of_dict):
    sorted_list = sorted(list_of_dict, key=itemgetter("cost"))
    return sorted_list


def clone_cell(
    parameter_dict, parent, dataset, model, prediction, target_vals
):
    clones = list()
    clones.append(parent)
    if parameter_dict["sort_by"] == "cost":
        normalized_cost = parent["norm_cost"]
    elif parameter_dict["sort_by"] == "distance":
        normalized_cost = parent["norm_cost"]
    else:
        print("fail on clone cell to pick sorting by cost/distance")
        breakpoint()
    for i in range(parameter_dict["num_clones"]):
        v = mutate(
            parameter_dict, parent, normalized_cost, dataset, model, prediction
        )
        child = create_mutated_cell(
            parameter_dict,
            v,
            normalized_cost,
            model,
            prediction,
            dataset,
            target_vals,
        )
        if child["distance"] == 0:
            print("child == fact")
            breakpoint()
        clones.append(child)
    if parameter_dict["sort_by"] == "cost":
        clones = sort_by_cost(clones)
    elif parameter_dict["sort_by"] == "distance":
        clones = sort_by_dist(clones)
    if len(clones) > 0:
        gc.collect()
        return clones[0]
    else:
        gc.collect()
        return parent


def calculate_normalized_cost(pop):
    # change to normalized_cost for german currently all cost between 0.5 and 1
    # therefore subtract 1/2 then * 2
    last = len(pop) - 1
    pop = sorted(pop, key=itemgetter("cost"))
    rg = pop[last][
        "cost"
    ]  # - pop[0]['cost'] #r changed from range to avoid confusion in loops
    if rg == 0:
        for cell in pop:
            cell["norm_cost"] = cell["cost"]
    else:
        for cell in pop:
            cell["norm_cost"] = 1 - (cell["cost"] / rg)
    return pop


def calculate_normalized_distance(pop):
    for p in pop:
        if p == None:
            pop.remove(p)
    last = len(pop) - 1
    pop = sorted(pop, key=itemgetter("distance"))
    rg = pop[last][
        "distance"
    ]  # - pop[0]['cost'] #r changed from range to avoid confusion in loops
    if rg == 0:
        for cell in pop:
            cell["norm_cost"] = cell["distance"]
    else:
        for cell in pop:
            cell["norm_cost"] = 1 - (cell["distance"] / rg)
    return pop


def average_cost(pop):
    sum = 0.0
    for cell in pop:
        sum = sum + cell["cost"]
    return sum / len(pop)


def average_distance(pop):
    sum = 0.0
    for cell in pop:
        sum = (
            sum + cell["distance"]
        )  # changed to distance as metric for mutation
    return sum / len(pop)


def get_offset(
    class_name, columns
):  # gets an offset depending if the class name is in class name or not
    if class_name in columns:  # assumes class_name is first in columns
        offset = 1
    else:
        offset = 0
    return offset


def combined_distance(parameter_dict, reference_cell, cell, dataset):
    # distance = 0
    # count = 0
    columns = get_line_columns(dataset)  # dataset['columns']
    d_array = []
    d_cont_arr = []
    d_cat_arr = []
    for i in range(len(cell)):
        if columns[i] in dataset["continuous"]:  # abs distance
            ref_val = reference_cell[i]
            cell_val = cell[i]
            dist = abs(ref_val - cell_val)
            d_cont_arr = np.append(d_cont_arr, dist)

        else:  # columns[i] in dataset['non_continuous']:#hamming distance
            ref_val = reference_cell[i]
            cell_val = cell[i]
            # if using dummies a change of 1 category results in two changes a change of 1 to 0 and a change of 0 to 1 this means that we need to change the cost of a change from 1 to 0.5
            if dataset["use_dummies"]:
                if ref_val == cell_val:
                    dist = 0
                else:
                    dist = 0.5
            else:
                if ref_val == cell_val:
                    dist = 0
                else:
                    dist = 1
            # change to using normalised hamming distance
            d_cat_arr = np.append(d_cat_arr, dist)

    # if not using MAD
    # for use when not diving in defintion of aff_thresh
    # get mad of continuous distances
    d_cont = 0
    d_cat = 0
    if parameter_dict["use_mads"] == True:
        mad = stats.median_abs_deviation(d_cont_arr, axis=None)
        sum_of_cat = np.sum(d_cat_arr)
        # distance for cat variables = 1/number of cat (sum of (abs distances/mad)
        for i in range(len(d_cont_arr)):
            if mad != 0:
                d_cont = d_cont + (abs(d_cont_arr[i]) / mad)
            else:
                d_cont = d_cont + abs(d_cont_arr[i])

        if len(d_cont_arr > 0):  # avoid /0 error
            d_cont = d_cont / len(d_cont_arr)
        for i in range(len(d_cat_arr)):
            d_cat = d_cat + d_cat_arr[i]
        if len(d_cat_arr) > 0:  # avoid /0 error
            d_cat = d_cat / len(d_cat_arr)
        # adjust for lengths of arrays for continuous and categorical features
        d_cont = d_cont * (
            len(d_cont_arr) / (len(d_cont_arr) + len(d_cat_arr))
        )
        d_cat = d_cat * (len(d_cat_arr) / (len(d_cont_arr) + len(d_cat_arr)))

        return d_cont + d_cat

    else:  # no mads
        for i in range(len(d_cont_arr)):
            d_cont = d_cont + abs(d_cont_arr[i])
        if len(d_cont_arr > 0):  # avoid /0 error
            d_cont = d_cont / len(d_cont_arr)
        for i in range(len(d_cat_arr)):
            d_cat = d_cat + d_cat_arr[i]
        if len(d_cat_arr) > 0:  # avoid /0 error
            d_cat = d_cat / len(d_cat_arr)
        # adjust for lengths of arrays for continuous and categorical features
        d_cont = d_cont * (
            len(d_cont_arr) / (len(d_cont_arr) + len(d_cat_arr))
        )
        d_cat = d_cat * (len(d_cat_arr) / (len(d_cont_arr) + len(d_cat_arr)))
        return d_cont + d_cat


def get_neighborhood(parameter_dict, reference_cell, pop, aff_thresh, dataset):
    neighbors = list()
    neighbors.append(reference_cell)  # add refencecell to neighbors
    # remove reference cell from pop
    del pop[0]
    out_pop = list()  # cells not in neighbourhood 2013.08.21
    lol = []
    for cell in pop:
        # if distance of population cell less than aff_thresh from reference cell add to neighbourhood
        # should this be add if more than aff_thresh?
        dist = combined_distance(
            parameter_dict,
            reference_cell["value"][0],
            cell["value"][0],
            dataset,
        )
        print(f"combined_distance: {dist}")
        lol.append(
            combined_distance(
                parameter_dict,
                reference_cell["value"][0],
                cell["value"][0],
                dataset,
            )
        )
        print(f"aff_thresh: {aff_thresh}")
        if (
            combined_distance(
                parameter_dict,
                reference_cell["value"][0],
                cell["value"][0],
                dataset,
            )
            < aff_thresh
        ):
            neighbors.append(cell)
        else:
            out_pop.append(cell)

    print(
        f"neighbors: {len(neighbors)}, out_pop: {len(out_pop)} pop: {len(pop)}"
    )
    # NOTE: I want larger out_pop and less neighbors
    return neighbors, out_pop


def affinity_suppress(parameter_dict, population, aff_thresh, dataset):
    out_pop = (
        []
    )  # holds population of cells that are first in their neighbourhoods
    flag = True
    while flag:
        new_list = list()
        first = population[0]
        neighbors, population = get_neighborhood(
            parameter_dict, first, population, aff_thresh, dataset
        )  # cell becomes population [i] 2023.08.21
        sorted_neighbors = sort_by_dist(
            neighbors
        )  # changed suppression metric from cost to distance
        out_pop.append(sorted_neighbors[0])

        if len(population) == 0:
            flag = False
    return out_pop


def affinity_suppress_by_cost(parameter_dict, population, aff_thresh, dataset):
    pop = (
        []
    )  # holds population of cells that are first in their neighbourhoods
    flag = True
    # population enters method sorted by distance sort by cost instead
    population = sort_by_cost(population)
    while flag:
        new_list = list()
        first = population[0]
        neighbors = get_neighborhood(
            parameter_dict, first, population, aff_thresh, dataset
        )
        sorted_neighbors = sort_by_cost(
            neighbors
        )  # changed suppression metric from cost to distance

        if len(sorted_neighbors) > 0:
            base_d = sorted_neighbors[0]["cost"]
            top_d = sorted_neighbors[len(sorted_neighbors) - 1]["cost"]
            pop.append(sorted_neighbors[0])

            for p in population:
                if p["cost"] > top_d or p["cost"] < base_d:
                    new_list.append(p)

        population = list()
        population = new_list

        if len(population) == 0:
            flag = False
    return pop


def descale_decode(line, columns, dataset):
    df_new_line = pd.DataFrame(columns=columns, index=[0])
    df_line = pd.DataFrame(data=line, columns=columns)
    for col in columns:
        if col in dataset["continuous"]:
            if dataset["scaler"] != "":
                sc = dataset["scaler"][col]
                df_new_line[col] = sc.inverse_transform(
                    df_line[col].values.reshape(1, -1)
                )
        if col in dataset["non_continuous"]:
            if col in dataset["label_encoder"]:
                le = dataset["label_encoder"][col]
                df_new_line[col] = le.inverse_transform(
                    df_line[col].values.astype(int)
                )
    return df_new_line


def search(
    parameter_dict, aff_thresh, dataset, model, prediction, target_vals
):
    print("begin search")
    num_of_att = len(dataset["columns"]) - len(dataset["invariants"]) - 1
    pop = []
    # col = []

    for i in range(parameter_dict["pop_size"]):
        # print('get new cell')
        cell = new_cell(
            parameter_dict, dataset, model, prediction, target_vals
        )
        if cell != None:
            pop.append(cell)
    if pop == []:
        return None, pd.DataFrame(columns=dataset["X_columns_with_dummies"])
    best = None
    print("initial pop size: :", len(pop))
    for g in range(parameter_dict["max_gens"]):
        print("Generation: ", g, " of ", parameter_dict["max_gens"])

        gc.collect()
        pop = [p for p in pop if p is not None]  # Remove None values safely
        for p in pop:
            if p != None:
                p["cost"] = objective_function(p["value"], model, prediction)
            # else:
            # pop.remove(p)
        calculate_normalized_distance(pop)
        calculate_normalized_cost(pop)
        pop = sort_by_cost(pop)

        print(f"This is len pop: {len(pop)}")

        if best == None:
            best = pop[0]
        else:
            if pop[0]["distance"] < best["distance"]:
                best = pop[0]
        if parameter_dict["sort_by"] == "distance":
            avgCost, progeny = average_distance(pop), None
        elif parameter_dict["sort_by"] == "cost":
            avgCost, progeny = average_cost(pop), None
        else:
            print("failure of global sort_by variable")
            breakpoint()
        flag = True
        avg_flag_count = 0
        while flag:
            progeny = list()
            for p in pop:
                cell = clone_cell(
                    parameter_dict, p, dataset, model, prediction, target_vals
                )
                progeny.append(cell)
            print(f"This is the length progeny {len(progeny)}")
            test = [x for x in progeny if x is not None]

            print(f"This is the length test {len(test)}")
            if parameter_dict["sort_by"] == "distance":
                prog_avg_cost = average_distance(progeny)
            elif parameter_dict["sort_by"] == "cost":
                prog_avg_cost = average_cost(progeny)
            else:
                print("failure of global sort_by variable in cloning")
                breakpoint()
            print("testing avg cost i lower")
            if prog_avg_cost < avgCost:
                flag = False
            if avg_flag_count > 9:  # escape from this loop
                flag = False
                print("escaping testing avg cost i lower")
            avg_flag_count += 1

        if num_of_att > 1:
            sorted_progeny = affinity_suppress(
                parameter_dict, progeny, aff_thresh, dataset
            )
            if parameter_dict["sort_by"] == "distance":
                sorted_progeny = sort_by_dist(sorted_progeny)
            elif parameter_dict["sort_by"] == "cost":
                sorted_progeny = sort_by_cost(sorted_progeny)
            else:
                print("failure of global sort_by variable in suppression")

        else:
            # IS THIS WRONG OR JUST NEVER CALLED
            print(
                "this branch should never be called subject to further investigation"
            )
            breakpoint()
            # if only one attribute and that attribute is non_continuous pointless sorting by distance as all are equally far apart
            columns = dataset["X_columns"]
            for col in columns:
                if columns in dataset["invariants"]:
                    columns.remove(col)

            if columns[0] in dataset["non_continuous"]:
                sorted_progeny = affinity_suppress_by_cost(
                    parameter_dict, progeny, aff_thresh, dataset
                )
                sorted_progeny = sort_by_cost(sorted_progeny)
            else:
                sorted_progeny = affinity_suppress(
                    parameter_dict, progeny, aff_thresh, dataset
                )
                sorted_progeny = sort_by_dist(sorted_progeny)

        best = sorted_progeny[0]
        stop_cost = 0

        for i in range(len(sorted_progeny)):
            stop_cost = stop_cost + sorted_progeny[i]["distance"]
        stop_cost = stop_cost / (
            parameter_dict["problem_size"] * len(sorted_progeny)
        )
        print(
            "stop_cost: ",
            stop_cost,
            "stop_cost/problem_size: ",
            stop_cost / (parameter_dict["problem_size"] * len(sorted_progeny)),
        )
        print("pop size after suppression: ", len(sorted_progeny))
        if stop_cost < parameter_dict["stop_condition"]:
            print("break because |cost| < ", parameter_dict["stop_condition"])
            break
        if (
            g < parameter_dict["max_gens"] - 2
        ):  # (max_gens/2 -1)this inserts random cells in evry generation, other term stops adding cells halfway through and just optimises no cells added in last generation
            new_cell_num = int(
                parameter_dict["pop_size"] * parameter_dict["new_cell_rate"]
            )  # get initial population from parameter dict

            for i in range(new_cell_num):  # get from global

                if (
                    len(sorted_progeny) < parameter_dict["pop_size"]
                ):  # get initial pop from global
                    sorted_progeny.append(
                        new_cell(
                            parameter_dict,
                            dataset,
                            model,
                            prediction,
                            target_vals,
                        )
                    )

        pop = sorted_progeny

    out_pop = list()
    with open("opt-AINet_results.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        kp = keras_pred(model, pop[0]["value"][0].reshape(1, -1))
        correct = False
        if prediction == 0:
            if kp[1] > kp[0]:
                correct = True
        else:
            if kp[0] > kp[1]:
                correct = True
        row = (prediction, kp[0], kp[1], correct)
        writer.writerow(row)
    for cell in sorted_progeny:
        temp_values = cell["value"][0].reshape(1, -1)
        line_columns = get_line_columns(dataset)
        temp_values = pd.DataFrame(
            data=temp_values[0].reshape(1, -1), columns=line_columns
        )  # columns = dataset['X_columns'])
        values = list()

        for col in line_columns:  # dataset['X_columns']:
            k = col
            v = temp_values[col].values[0]
            value = {k: v}
            values.append(value)

        cost = cell["cost"]
        distance = cell["distance"]
        entry = {"values": values, "cost": cost, "distance": distance}
        out_pop.append(entry)
    best = out_pop[0]
    ais_columns = list()
    line_columns = get_line_columns(dataset)
    dataset["X_columns_with_dummies"]

    for col in line_columns:
        ais_columns.append(col)

    ais_columns.append("distance")
    ais_columns.append("cost")
    df_pop = pd.DataFrame(columns=ais_columns)
    for cell in sorted_progeny:
        temp_values = cell["value"][0].reshape(1, -1)
        temp_values = np.append(temp_values, cell["distance"])
        temp_values = np.append(temp_values, cell["cost"])
        df_temp = pd.DataFrame(
            data=temp_values.reshape(1, -1), columns=ais_columns
        )
        df_pop = pd.concat([df_pop, df_temp])
    print(df_pop)
    return best, df_pop


def create_optAINet_explanation(
    target_vals, y, pop, dataset, prob_dict, record_number, db_file, imp_arr
):

    description = "optAINet"
    conn = create_connection(db_file)

    with conn:
        explanations_id = write_to_db.db_add_explanations(
            conn, description, record_number
        )
        records_id = write_to_db.db_add_records(conn, explanations_id)
        # add record all operands will be '='
        # add class
        class_name = dataset["class_name"]
        class_value = str(y)
        operand = "="
        attributes_id = write_to_db.db_add_attributes_bundle(
            conn, class_name, operand, y
        )
        records_attributes_id = write_to_db.db_add_records_attributes(
            conn, attributes_id, records_id
        )

        # in this section create section for if using dummies
        # add all non class columns
        # ***************ADD RECORD VALUES*************
        if dataset["use_dummies"] == True:

            for col in dataset["X_columns"]:
                att_type = col
                operand = "="
                att_value = ""
                if col in dataset["continuous"]:
                    att_value = str(target_vals[col][0])

                else:
                    for sub_col in dataset["dummy"][col]:
                        if target_vals[sub_col][0] == 1:
                            att_value = sub_col

                att_value = write_to_db.decode_debin_db(
                    att_type, att_value, dataset
                )
                if dataset["data_human_dict"] != {}:
                    if att_type in dataset["continuous"]:
                        att_type = dataset["data_human_dict"][att_type]
                    else:
                        att_type = dataset["data_human_dict"][att_value][
                            "name"
                        ]
                        att_value = dataset["data_human_dict"][att_value][
                            "value"
                        ]

                attributes_id = write_to_db.db_add_attributes_bundle(
                    conn, att_type, operand, att_value
                )
                records_attributes_id = write_to_db.db_add_records_attributes(
                    conn, attributes_id, records_id
                )

            # add importances
            importance_id = write_to_db.db_add_importances(
                conn, explanations_id
            )
            for i in range(len(imp_arr)):
                line_columns = get_line_columns(dataset)
                att_type = line_columns[i]
                operand = "="
                att_value = imp_arr[i]
                attribute_id = write_to_db.db_add_attributes_bundle(
                    conn, att_type, operand, att_value
                )
                importances_attributes_id = (
                    write_to_db.db_add_importances_attributes(
                        conn, importance_id, attribute_id
                    )
                )
        else:
            for col in dataset["X_columns"]:
                att_type = col
                operand = "="
                att_value = int(target_vals[col][0])
                att_value = write_to_db.decode_debin_db(
                    att_type, att_value, dataset
                )
                attributes_id = write_to_db.db_add_attributes_bundle(
                    conn, att_type, operand, att_value
                )
                records_attributes_id = write_to_db.db_add_records_attributes(
                    conn, attributes_id, records_id
                )
            # add importances
            importance_id = write_to_db.db_add_importances(
                conn, explanations_id
            )
            for i in range(len(imp_arr)):
                att_type = dataset["X_columns"][i]
                operand = "="
                att_value = imp_arr[i]
                attribute_id = write_to_db.db_add_attributes_bundle(
                    conn, att_type, operand, att_value
                )
                importances_attributes_id = (
                    write_to_db.db_add_importances_attributes(
                        conn, importance_id, attribute_id
                    )
                )

        # add invariant columns to db
        for col in dataset["invariants"]:
            att_type = col
            operand = "="
            att_value = ""
            attributes_id = write_to_db.db_add_attributes_bundle(
                conn, att_type, operand, att_value
            )
            invariants_attributes_id = db_add_invariants_attributes(
                conn, attributes_id, records_id
            )

        # ***************Add CFs***************
        for i in range(
            len(pop)
        ):  # change to lesser of len(pop) or X, suggested value of X = 5

            deltas_id = write_to_db.db_add_deltas(conn, explanations_id)
            operand = "="
            att_name = ""
            att_value = ""

            for col in pop.columns:
                att_name = col  # key for attributes_deltas
                # cost and distance cannot be an int
                line_columns = get_line_columns(dataset)
                if col in line_columns:
                    if dataset["use_dummies"] == True:
                        if col in dataset["continuous"]:

                            att_value = pop[col].values[i]

                            att_value = write_to_db.decode_debin_db(
                                col, att_value, dataset, decode=True
                            )
                            att_name = dataset["data_human_dict"][col]
                            attributes_id = (
                                write_to_db.db_add_attributes_bundle(
                                    conn, att_name, operand, att_value
                                )
                            )
                            attributes_deltas_id = (
                                write_to_db.db_add_deltas_attributes(
                                    conn, deltas_id, attributes_id
                                )
                            )

                        else:
                            for dummied_col in dataset["dummy"]:
                                for dummy_value in dataset["dummy"][
                                    dummied_col
                                ]:
                                    if (
                                        int(pop[dummy_value].values[i]) == 1
                                    ) and (col == dummy_value):
                                        att_name = dummied_col
                                        att_value = dataset["data_human_dict"][
                                            col
                                        ]["value"]
                                        att_name = dataset["data_human_dict"][
                                            col
                                        ]["name"]

                                        att_value = (
                                            write_to_db.decode_debin_db(
                                                col,
                                                att_value,
                                                dataset,
                                                decode=True,
                                            )
                                        )
                                        attributes_id = write_to_db.db_add_attributes_bundle(
                                            conn, att_name, operand, att_value
                                        )
                                        attributes_deltas_id = write_to_db.db_add_deltas_attributes(
                                            conn, deltas_id, attributes_id
                                        )

                    else:
                        if col in line_columns:  # ie not distance or cost
                            att_value = int(pop.iloc[i][col])
                            # with all decoding for opt-AINet being done values should need decoding now changing decode to True
                            att_value = write_to_db.decode_debin_db(
                                col, att_value, dataset, decode=True
                            )

                else:
                    # distance and cost
                    # distance and cost must not be transformed to ints
                    att_value = str(pop.iloc[i][col])

                    attributes_id = write_to_db.db_add_attributes_bundle(
                        conn, att_name, operand, att_value
                    )
                    attributes_deltas_id = (
                        write_to_db.db_add_deltas_attributes(
                            conn, deltas_id, attributes_id
                        )
                    )

        for key in prob_dict:
            class_probabilities_id = write_to_db.db_add_class_probabilities(
                conn, explanations_id
            )
            operand = "="
            attributes_id = write_to_db.db_add_attributes_bundle(
                conn, key, operand, prob_dict[key]
            )
            class_probabilities_attributes_id = (
                write_to_db.db_add_class_probabilities_attributes(
                    conn, class_probabilities_id, attributes_id
                )
            )


def get_invariants(dataset, num_variants, X, y):
    # num_variants = 10
    lr = LinearRegression()
    lr = lr.fit(X, y)
    coeffs = lr.coef_
    coeff_dict = {}
    i = 0
    for col in dataset["X_columns"]:
        coeff_dict[col] = abs(coeffs[i])
        i = i + 1
    # sort coeff_dict by abs magnitude
    sorted_coeff_dict = sorted(
        coeff_dict.items(), reverse=True, key=lambda x: x[1]
    )
    # loop all invariants all outside top 5
    invariant_list = list()

    for i in range(num_variants, X.shape[1]):
        invariant_list.append(sorted_coeff_dict[i][0])
    return invariant_list

    line_columns = dataset["X_columns"]
    if dataset["use_dummies"] == True:
        line_columns = dataset["X_columns_with_dummies"]
    return line_columns


def init_var_optAINet(
    blackbox,
    X,
    line_number,
    dataset,
    prob_dict,
    db_file,
    imp_arr,
    df_out,
    parameter_dict,
):
    def predict_label(y):
        if y < 0.5:
            return 0
        else:
            return 1

    # prediction = predict_label(blackbox.predict(X[line_number].reshape(1, -1))) #old line
    prediction = predict_label(
        predict.predict_single(blackbox, X[line_number])
    )
    print(
        f"This is pred of query instance {predict.predict_single(blackbox, X[line_number])}"
    )
    # if outside of a notebook set parameter dict
    if parameter_dict == {}:
        parameter_dict = set_parameter_dict()

    line = X[line_number]  # np rows
    print(f"This is the type {type(X[line_number])}")
    aff_thresh = (
        (parameter_dict["search_space"][1] - parameter_dict["search_space"][0])
        / parameter_dict["affinity_constant"]
    ) / (1 + len(dataset["X_columns"]) - len(dataset["invariants"]))

    print(f"aff_thresh: {aff_thresh}")
    best, df_pop = search(
        parameter_dict, aff_thresh, dataset, blackbox, prediction, line
    )

    # print(df_pop)
    if best == None:
        print("no cell created for this run")
        return (
            False,
            df_pop,
        )  # False for no cell with these invariant attributes
    # if df_pop == []:return False,df_out

    if df_out.shape[0] == 0:
        df_out = df_pop
    else:
        # df_pop = df_out.append(df_pop)
        df_pop = pd.concat((df_out, df_pop))
    line_cost = keras_pred(blackbox, line.reshape(1, -1))

    # add 0, 0 for distance and cost to line
    ais_line = copy.deepcopy(line)
    ais_line = np.append(ais_line, 0)  # 0 distance to itself
    ais_line = np.append(ais_line, line_cost[1])
    ais_columns = list()
    line_columns = get_line_columns(dataset)
    # for col in dataset['X_columns']:
    for col in line_columns:
        ais_columns.append(col)
    # ais_columns = (copy.deepcopy(dataset['X_columns']))
    ais_columns.append("distance")
    ais_columns.append("cost")

    df_line = pd.DataFrame(data=ais_line.reshape(1, -1), columns=ais_columns)
    print("length of invariants: ", len(dataset["invariants"]))

    # create_optAINet_explanation(
    #     df_line,
    #     dataset["possible_outcomes"][prediction],
    #     df_out,
    #     dataset,
    #     prob_dict,
    #     line_number,
    #     db_file,
    #     imp_arr,
    # )

    if len(dataset["invariants"]) > 0:
        return False, df_out
    else:
        return (
            True,
            df_out,
        )  # True for cell created with these invariant attributes
