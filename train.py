import util as ut
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
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
    clf = GridSearchCV(model, params, cv=10)
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
    precision, recall, fscore, support = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    accuracy = accuracy_score(Y_true, Y_pred)

    return accuracy, precision, recall, accuracy, fscore


def train_model(X_train, Y_train, choice, params):
    """
    Description: General function for building model specified by "model" param.
    :param X_train: pandas training data frame
    :param Y_train: pandas Series with true labels
    :param choice: Choice of machine learning model to be used for training.
    :param params: Dictionary containing range of parameter values for GridSearch hyper parameter tuning.
    :return: Trained model object.
    """
    if choice == "DT":
        model = DecisionTreeClassifier()
    elif choice == "SGD":
        model = SGDClassifier()
    elif choice == "MLP":
        model = MLPClassifier()
    elif choice == "RF":
        model = RandomForestClassifier(max_features='sqrt')
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

    params = {'criterion': ['gini', 'entropy'], 'min_samples_split': [0.01, 0.05, 0.1],
              'min_samples_leaf': [0.001, 0.005, 0.01], 'max_features': ["sqrt", 1.0]}
    cv_model = train_model(X_train, Y_train, "DT", params)
    ut.write_pickle_file(cv_model, './output/dt_model.pkl')

    params = {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # learning rate
        'max_iter': [5000, 10000],  # number of epochs
        'loss': ['modified_huber'],
        'penalty': ['l2'],
        'tol': ['1e-6', '1e-3', '1e-4', '1e-5']
    }
    cv_model = train_model(X_train, Y_train, "SGD", params)
    ut.write_pickle_file(cv_model, './output/sgd_model.pkl')

    params = {'solver': ['adam', 'sgd'],
              'activation': ['logistic', 'relu', 'tanh'],
              'alpha': [0.001, 0.05, 0.5],
              'tol': ['1e-6', '1e-3', '1e-4', '1e-5'],
              'hidden_layer_sizes': [(6, 20), (100, 20)],
              'max_iter': [100, 200, 500]
              }
    cv_model = train_model(X_train, Y_train, "MLP", params)
    ut.write_pickle_file(cv_model, './output/mlp_model.pkl')

    params = {'n_estimators': [100, 500, 1000, 10000], 'min_samples_split': [0.01, 0.05, 0.1], 'min_samples_leaf': [0.001, 0.005, 0.01]}
    cv_model = train_model(X_train, Y_train, "RF", params)
    ut.write_pickle_file(cv_model, './output/rf_model.pkl')

    params = {'n_estimators': [100, 500, 1000, 10000], 'loss': ['deviance', 'exponential'],
              'learning_rate': [0.01, 0.05, 0.1, 0.5], 'subsample': [0.9, 0.85]}
    cv_model = train_model(X_train, Y_train, "GB", params)
    ut.write_pickle_file(cv_model, './output/gb_model.pkl')