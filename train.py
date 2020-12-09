import util as ut
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


def tune_hyper_params(model, params, X_train, Y_train):
    """
    Description: Function for hyperparameter tuning based on cross validation and gridsearch
    :param model:  sklearn model whose hyper parameters need to be tuned based on cross validation
    :param params: A dictionary with keys being the hyperparameter and the value being the list of
                    parameters that need to explored
    :param X_train: pandas training data frame
    :param Y_train: pandas Series with true labels
    :return: The best estimator based on hyper parameter tuning
    """
    clf = GridSearchCV(model, params, cv=5, verbose=1, n_jobs=-1)
    clf.fit(X_train, Y_train)
    return clf


def evaluate_model(model, X_test, Y_true):
    """
    Description: Function for evaluating model performance on test data
    :param model:  sklearn model that needs to be testes
    :param X_test: test data
    :param Y_true: true labels corresponding to the test data
    :return: accuracy, precision, recall, accuracy, fscore of the estimator on test data
    """
    Y_pred = model.predict(X_test)
    # precision, recall, fscore, support = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    # accuracy = accuracy_score(Y_true, Y_pred)
    metric_table = classification_report(Y_true, Y_pred)
    return metric_table

def get_class_weight_dict(Y_train):
    """
       Description: Assign class weights to outcome variable
       :param Y_train: pandas Series corresponding to label
       :return: A dictionary with class weights assigned
    """
    uq_values = Y_train.value_counts(sort=False).to_dict()
    n = Y_train.shape[0]
    for key, val in uq_values.items():
        uq_values[key] = uq_values[key] / (1.0 * n)
    return uq_values

def train_model(X_train, Y_train, choice, params):
    """
    Description: General function for building model specified by "model" param.
    :param X_train: pandas training data frame
    :param Y_train: pandas Series with true labels
    :param choice: Choice of machine learning model to be used for training.
    :param params: Dictionary containing range of parameter values for GridSearch hyper parameter tuning.
    :return: Trained model object.
    """
    class_weights = get_class_weight_dict(Y_train)

    if choice == "DT":
        model = DecisionTreeClassifier(class_weight=class_weights)
    elif choice == "SGD":
        model = SGDClassifier(class_weight=class_weights)
    elif choice == "MLP":
        model = MLPClassifier()
    elif choice == "RF":
        model = RandomForestClassifier(max_features='sqrt', class_weight=class_weights)
    elif choice == "GB":
        model = GradientBoostingClassifier(max_features='sqrt')
    else:
        model = DummyClassifier()
    model = tune_hyper_params(model, params, X_train, Y_train)
    return model


if __name__ == "__main__":
    '''
    Description: Test file to train and evaluate models
    '''
    np.random.seed(10)
    train_df, test_df = ut.load_train_and_test_data('./output')
    X_train, Y_train = train_df.drop(['video_id', 'label'], axis=1), train_df['label'].astype(str)
    X_test, Y_test = test_df.drop(['video_id', 'label'], axis=1), test_df['label'].astype(str)

    params = {}
    cv_model = train_model(X_train, Y_train, "baseline", params)
    ut.write_pickle_file(cv_model, './output/baseline_model.pkl')
    print("Model baseline built")

    params = {'criterion': ['gini', 'entropy'], 'min_samples_split': [0.01, 0.05, 0.1],
              'min_samples_leaf': [0.001, 0.005, 0.01], 'max_features': ["sqrt", 1.0]}

    cv_model = train_model(X_train, Y_train, "DT", params)
    ut.write_pickle_file(cv_model, './output/dt_model.pkl')
    print("Model DT built")
    params = {
        'alpha': [1e-5, 1e-4, 1e-3],  # learning rate
        'max_iter': [5000, 10000],  # number of epochs
        'loss': ['modified_huber'],
        'penalty': ['l2', 'l1'],
        'tol': [1e-5, 1e-3, 1e-4]
    }

    cv_model = train_model(X_train, Y_train, "SGD", params)
    ut.write_pickle_file(cv_model, './output/sgd_model.pkl')
    print("Model SGD built")

    params = {'solver': ['adam', 'sgd'],
              'activation': ['logistic', 'relu', 'tanh'],
              'alpha': [0.001, 0.05, 0.5],
              'tol': [1e-5, 1e-3, 1e-4],
              'hidden_layer_sizes': [(6, 20), (100, 20)],
              'max_iter': [100, 200, 500]
              }
    cv_model = train_model(X_train, Y_train, "MLP", params)
    ut.write_pickle_file(cv_model, './output/mlp_model.pkl')
    print("Model MLP built")

    params = {'n_estimators': [100, 500, 1000, 10000], 'min_samples_split': [0.01, 0.05, 0.1], 'min_samples_leaf': [0.001, 0.005, 0.01]}

    cv_model = train_model(X_train, Y_train, "RF", params)
    ut.write_pickle_file(cv_model, './output/rf_model.pkl')
    print("Model RF built")

    params = {'n_estimators': [100, 500, 1000, 10000], 'loss': ['deviance', 'exponential'],
              'learning_rate': [0.01, 0.05, 0.1, 0.5], 'subsample': [0.9, 0.85]}
    cv_model = train_model(X_train, Y_train, "GB", params)
    ut.write_pickle_file(cv_model, './output/gb_model.pkl')
    print("Model GB built")