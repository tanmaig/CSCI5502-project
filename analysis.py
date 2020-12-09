import pandas as pd
import argparse
from util import read_pickle_file
from train import evaluate_model
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        default="./output",
                        help="Path to output folder.")

    args = parser.parse_args()
    output_path = args.output_path

    baseline_cv_model = read_pickle_file(output_path + "/baseline_model.pkl")
    decision_tree_cv_model = read_pickle_file(output_path + "/dt_model.pkl")
    sgd_cv_model = read_pickle_file(output_path + "/sgd_model.pkl")
    mlp_cv_model = read_pickle_file(output_path + "/mlp_model.pkl")
    rf_cv_model = read_pickle_file(output_path + "/rf_model.pkl")
    gb_cv_model = read_pickle_file(output_path + "/gb_model.pkl")

    # Best models
    baseline_model = baseline_cv_model.best_estimator_
    dt_model = decision_tree_cv_model.best_estimator_
    sgd_model = sgd_cv_model.best_estimator_
    mlp_model = mlp_cv_model.best_estimator_
    rf_model = rf_cv_model.best_estimator_
    gb_model = gb_cv_model.best_estimator_

    # Read test data
    test = read_pickle_file(output_path + "/test.pkl")
    X_test = test[test.columns[~test.columns.isin(["label"])]]
    y_test = test["label"]

    # Predict on test data.
    results = {"Baseline": evaluate_model(baseline_model, X_test, y_test),
               "Decision Tree": evaluate_model(dt_model, X_test, y_test),
               "Stochastic Gradient Descent": evaluate_model(sgd_model, X_test, y_test),
               "Multi-Layer Perceptron": evaluate_model(sgd_model, X_test, y_test),
               "Random Forrest": evaluate_model(sgd_model, X_test, y_test),
               "Gradient Boosting": evaluate_model(sgd_model, X_test, y_test)}

    for model, metrics_table in results.items():
        print("Results for " + model + ":")
        print(metrics_table)