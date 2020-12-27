import numpy as np

from lab4_utils import feature_names, accuracy_score


def replace_mode(inputs, labels):
    # print("\nMODE VALUES:")
    # replace missing inputs with mode
    for input in inputs:

        # Mode Values
        '''
        age = stats.mode(input[:, 0])[0]
        menopause = stats.mode(input[:, 1])[0]
        tumor_size = stats.mode(input[:, 2])[0]
        inv_nodes = stats.mode(input[:, 3])[0]
        node_caps = stats.mode(input[:, 4])[0]
        deg_malig = stats.mode(input[:, 5])[0]
        breast = stats.mode(input[:, 6])[0]
        b_quad = stats.mode(input[:, 7])[0]
        irradiant = stats.mode(input[:, 8])[0]
        '''

        # mode = [age, menopause, tumor_size, inv_nodes, node_caps, deg_malig, breast, b_quad, irradiant]
        # print(age, menopause, tumor_size, inv_nodes, node_caps, deg_malig, breast, b_quad, irradiant)

        mode = []
        for i in range(9):
            (data, counts) = np.unique(input[:, i], return_counts=True)
            # print(data, idx, counts)
            index = np.argmax(counts)
            # print(np.argmax(counts))
            result = data[index]
            mode.append(result)
            # print(result)
        # print(mode)

        for i in range(9):
            category = input[:, i]
            unique = np.unique(category)
            if '?' in unique:
                # print("? FOUND IN COLUMN " + str(i))
                category[category == '?'] = mode[i]  # replace ? with mode value

    # replace missing labels with mode

    for label in labels:
        (data, counts) = np.unique(label, return_counts=True)
        index = np.argmax(counts)
        mode = data[index]
        # mode.append(result)
        # mode = stats.mode(label)[0]
        label[label == '?'] = mode
        # print(mode)


# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])
    # VVVVV YOUR CODE GOES HERE VVVVV $

    inputs = (training_inputs, testing_inputs)
    labels = (training_labels, testing_labels)
    # REPLACE MISSING VALUES WITH MODE
    replace_mode(inputs, labels)
    processed_training_inputs = inputs[0]
    processed_testing_inputs = inputs[1]
    processed_training_labels = labels[0]
    processed_testing_labels = labels[1]
    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $

    return (
        processed_training_inputs,
        processed_testing_inputs,
        processed_training_labels,
        processed_testing_labels,
    )


def counts(data, keyword, column):
    count = 0
    for patient in data:
        # print(keyword, patient[column])
        if keyword == patient[column]:
            count += 1
    return count


# Hint: consider how to utilize np.argsort()
def naive_bayes(predict_on, reference_points, reference_labels):
    # Here you should calculate the requisite probabilities from the reference points
    # and then use them to classify each test point. Don't forget to use Laplace smoothing
    assert (
        len(predict_on) > 0
    ), f"parameter predict_on needs to be of length 0 or greater"
    assert (
        len(reference_points) > 0
    ), f"parameter reference_points needs to be of length 0 or greater"
    assert (
        len(reference_labels) > 0
    ), f"parameter reference_labels needs to be of length 0 or greater"
    assert len(reference_labels) == len(reference_points), (
        f"reference_points and reference_labels need to be the" f" same length"
    )
    predictions = []
    # VVVVV YOUR CODE GOES HERE VVVVV $

    # print(predict_on)
    # print(reference_points)
    # print(reference_labels)

    # age, meno, tumor, inv, node, deg, breast, quad, irradiat
    data_format = [["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
                   ["lt40", "ge40", "premeno"],
                   ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"],
                   ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"],
                   ["yes", "no"],
                   [1, 2, 3],
                   ["left", "right"],
                   ["left_up", "left_low", "right_up", "right_low", "central"],
                   ["yes", "no"]]
    feature_levels = [9, 3, 12, 13, 2, 3, 2, 5, 2]
    recurrence = []
    no_recurrence = []
    for i in range(len(reference_labels)):

        if reference_labels[i] == "recurrence-events":
            print(reference_points[i])
            recurrence.append(reference_points[i])
        else:
            print(reference_points[i])
            no_recurrence.append(reference_points[i])

    recurrence_prob = []
    no_recurrence_prob = []
    p_recurrence = len(recurrence) / (len(recurrence) + len(no_recurrence))
    print(p_recurrence)
    p_no_recurrence = len(no_recurrence) / (len(recurrence) + len(no_recurrence))

    # print(len(recurrence))

    for i in range(len(data_format)):
        results = []
        for j in range(len(data_format[i])):
            count = counts(recurrence, data_format[i][j], i)
            probability = (count + 1) / (len(recurrence) + len(data_format[i]))
            print(data_format[i][j], count, probability)
            node = (data_format[i][j], probability)
            results.append(node)
            # print(node)
        recurrence_prob.append(results)
        # print(results)

    for i in range(len(data_format)):
        results = []
        for j in range(len(data_format[i])):
            count = counts(no_recurrence, data_format[i][j], i)
            probability = (count + 1) / (len(no_recurrence) + len(data_format[i]))
            # print(data_format[i][j], count, probability)
            node = (data_format[i][j], probability)
            results.append(node)
        no_recurrence_prob.append(results)

    for patient in predict_on:
        rec_result = p_recurrence
        no_rec_result = p_no_recurrence
        for feature in range(len(patient)):
            # print(patient[feature])
            rec_p = [node for node in recurrence_prob[feature] if node[0] == patient[feature]]
            # print(rec_p[0][1])
            no_rec_p = [node for node in no_recurrence_prob[feature] if node[0] == patient[feature]]
            rec_result *= rec_p[0][1]
            no_rec_result *= no_rec_p[0][1]

        if (rec_result >= no_rec_result):
            predictions.append("recurrence-events")
        else:
            predictions.append("no-recurrence-events")


    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return predictions


def cross_validate(
    training_inputs, testing_inputs, training_labels, testing_labels, k=5
):
    # Here you should re-split the dataset into folds, preprocess again,
    #   run through your naive bayes, and measure the accuracy for each fold.
    # See the test script for examples of part of this process.

    kf_ete = []  # array of k-fold cross validation estimates of test error

    # VVVVV YOUR CODE GOES HERE VVVVV $

    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return kf_ete
