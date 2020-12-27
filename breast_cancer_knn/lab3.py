import numpy as np
# from scipy import stats
from lab3_utils import edit_distance, feature_names


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
    inputs = (training_inputs, testing_inputs)
    labels = (training_labels, testing_labels)

    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])

    # REPLACE MISSING VALUES WITH MODE
    # mode(training) -> training
    # mode(testing) -> testing
    replace_mode(inputs, labels)

    ages = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
    tumors = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"]
    nodes = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35",
             "36-39"]

    # ONE-HOT ENCODING
    for i in range(0, 2):

        # INPUTS
        for patient in inputs[i]:
            numerical_patient = []

            # AGE
            numerical_patient.append(ages.index(patient[0]) + 1)

            # MENOPAUSE
            if patient[1] == "lt40":
                numerical_patient.extend((1, 0, 0))
            elif patient[1] == "ge40":
                numerical_patient.extend((0, 1, 0))
            elif patient[1] == "premeno":
                numerical_patient.extend((0, 0, 1))

            # TUMOR SIZE
            numerical_patient.append(tumors.index(patient[2]) + 1)

            # INV NODES
            numerical_patient.append(nodes.index(patient[3]) + 1)

            # NODE CAPS
            if patient[4] == "no":
                numerical_patient.append(0)
            elif patient[4] == "yes":
                numerical_patient.append(1)

            # DEG MALIG
            numerical_patient.append(patient[5])

            # BREAST
            if patient[6] == "left":
                numerical_patient.append(0)
            elif patient[6] == "right":
                numerical_patient.append(1)

            # BREAST QUAD
            if patient[7] == "left_up":
                numerical_patient.extend((1, 0, 0, 0, 0))
            if patient[7] == "left_low":
                numerical_patient.extend((0, 1, 0, 0, 0))
            if patient[7] == "right_up":
                numerical_patient.extend((0, 0, 1, 0, 0))
            if patient[7] == "right_low":
                numerical_patient.extend((0, 0, 0, 1, 0))
            if patient[7] == "central":
                numerical_patient.extend((0, 0, 0, 0, 1))

            # IRRADIAL
            if patient[8] == "no":
                numerical_patient.append(0)
            if patient[8] == "yes":
                numerical_patient.append(1)

            if i == 0:
                processed_training_inputs.append(numerical_patient)
            elif i == 1:
                processed_testing_inputs.append(numerical_patient)

            # print(patient)
            # print(numerical_patient)

        # LABELS
        for event in labels[i]:
            # print(event)
            if i == 0 and event == "no-recurrence-events":
                processed_training_labels.append(0)
            elif i == 0 and event == "recurrence-events":
                processed_training_labels.append(1)
            elif i == 1 and event == "no-recurrence-events":
                processed_testing_labels.append(0)
            elif i == 1 and event == "recurrence-events":
                processed_testing_labels.append(1)

    '''
    print()
    print("Training Inputs")
    print("Size: " + str(len(training_inputs)))
    # for i in training_inputs:
        # print(i)
    print("Training Labels")
    print("Size: " + str(len(training_labels)))
    print("Testing Inputs")
    print("Size: " + str(len(testing_inputs)))
    print("Testing Labels")
    print("Size: " + str(len(testing_labels)))
    print()
    '''
    # for i in processed_testing_inputs:
        # print(i)
    # print(processed_testing_inputs)
    return processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels


# Hint: consider how to utilize np.argsort()
def k_nearest_neighbors(predict_on, reference_points, reference_labels, k, l, weighted):
    assert len(predict_on) > 0, f"parameter predict_on needs to be of length 0 or greater"
    assert len(reference_points) > 0, f"parameter reference_points needs to be of length 0 or greater"
    assert len(reference_labels) > 0, f"parameter reference_labels needs to be of length 0 or greater"
    assert len(reference_labels) == len(reference_points), f"reference_points and reference_labels need to be the" \
                                                           f" same length"
    predictions = []

    for predict in predict_on:
        # print(predict)
        distances = []
        for i in range(len(reference_points)):
            # print(reference_points[i])
            tuple = (float(edit_distance(reference_points[i], predict, l)), int(reference_labels[i]), reference_points[i])
            # tuple = (float(edit_distance(predict, reference_points[i], l)), int(reference_labels[i]), reference_points[i])
            distances.append(tuple)

        # print(distances)
        distances = sorted(distances, key=lambda x: (x[0], -x[1]))
        # sorted_distances = np.argsort(distances)
        # print(distances)

        if weighted:
            num_sum = 0
            denom_sum = 0
            for i in range(k):
                # print(distances[i][2])
                # print(distances[i][1], distances[i][0])
                num_sum += distances[i][1] / (distances[i][0] + 0.0001)
                denom_sum += 1 / (distances[i][0] + 0.0001)
            score = num_sum/denom_sum

        else:
            k_neighbors = []
            for i in range(k):
                result = distances[i][1]
                # print(result)
                k_neighbors.append(result)

            # predictions.append(stats.mode(k_neighbors))
            values, counts = np.unique(k_neighbors, return_counts=True)
            # print(counts)
            index = np.where(values == 1)
            count = counts[index]
            score = count / k

        if score >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
