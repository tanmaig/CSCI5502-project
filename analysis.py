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
    results = pd.DataFrame({"Model": ["Baseline",
                                      "Decision Tree",
                                      "Stochastic Gradient Descent",
                                      "Multi-Layer Perceptron",
                                      "Random Forrest",
                                      "Gradient Boosting"],
                            "metrics": [evaluate_model(baseline_model, X_test, y_test),
                                        evaluate_model(dt_model, X_test, y_test),
                                        evaluate_model(sgd_model, X_test, y_test),
                                        evaluate_model(mlp_model, X_test, y_test),
                                        evaluate_model(rf_model, X_test, y_test),
                                        evaluate_model(gb_model, X_test, y_test)]})

    results["accuracy"] = results.apply(lambda x: x["metrics"][0])
    results["precision"] = results.apply(lambda x: x["metrics"][1])
    results["recall"] = results.apply(lambda x: x["metrics"][2])
    results["support"] = results.apply(lambda x: x["metrics"][3])
    results["fscore"] = results.apply(lambda x: x["metrics"][4])
    results.drop(columns=["metrics"], inplace=True)