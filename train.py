import util as ut
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


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
    precision, recall, fscore, support  = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    accuracy = accuracy_score(Y_true, Y_pred)

    return accuracy, precision, recall, accuracy, fscore

def train_decision_tree(X_train, Y_train):
    """
       Description: Function for building DecisionTreeClassifier model
       :param X_train: pandas training data frame
       :param Y_train: pandas Series with true labels
       :return: sklearn DecisionTree model.
    """
    model = DecisionTreeClassifier()
    params = {'criterion': ['gini', 'entropy'], 'min_samples_split': [0.01, 0.05, 0.1], 'min_samples_leaf': [0.001, 0.005, 0.01], 'max_features': ["sqrt", 1.0]}
    model = tune_hyper_params(model, params, X_train, Y_train)
    return model

if __name__ == "__main__":
    '''
    Description: Test file to train and evaluate models
    '''
    np.random.seed(10)
    train_df, test_df = ut.load_train_and_test_data('./output')
    X_train, Y_train = train_df.drop(['label'], axis=1), train_df['label'].astype(str)
    X_test, Y_test = test_df.drop(['label'], axis=1), test_df['label'].astype(str)

    cv_model = train_decision_tree(X_train, Y_train)
    ut.write_pickle_file(cv_model, './output/dt_model.pkl')
