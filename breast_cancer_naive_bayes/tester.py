from lab4_utils import accuracy_score, load_data
from lab4 import naive_bayes, preprocess_data, cross_validate
import numpy as np
import time


if __name__ == "__main__":
    print("Loading data from file...")
    start_time = time.time()
    raw_data = load_data()

    load_time = time.time() - start_time

    print(f"Data loaded - time elapsed from start: {load_time:0.9f}")

    print("Beginning data preprocessing and cleaning...")
    (
        processed_training_inputs,
        processed_testing_inputs,
        processed_training_labels,
        processed_testing_labels,
    ) = preprocess_data(*raw_data)
    load_time = time.time() - start_time
    print(f"Data preprocessed - time elapsed from start: {load_time:0.9f}")

    print(
        f"Example training input: {processed_training_inputs[0]} - label: {processed_training_labels[0]}"
    )

    print(f"Running NB")
    predicted_labels = naive_bayes(
        predict_on=processed_testing_inputs,
        reference_points=processed_training_inputs,
        reference_labels=processed_training_labels,
    )
    load_time = time.time() - start_time
    print(f"NB completed - time elapsed from start: {load_time:0.9f}")

    accuracy = accuracy_score(processed_testing_labels, predicted_labels)
    print("Accuracy: " + str(accuracy))

    print(f"Running Cross Validation")
    kf_ete = cross_validate(*raw_data)

    load_time = time.time() - start_time
    print(f"Cross Validation completed - time elapsed from start: {load_time:0.9f}")
