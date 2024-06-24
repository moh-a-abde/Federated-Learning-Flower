import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import time

class XGBoostModel:
    def __init__(self, num_classes: int, input_dim: int):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None

    def train(self, X_train, y_train):
        # Transform numpy array into DMatrix format for xgboost to handle more efficiently
        train = xgb.DMatrix(X_train, label=y_train)

        # Parameters to train the model
        param = {
            'max_depth': 6,
            'eta': 0.35,
            'objective': 'multi:softmax',
            'num_class': self.num_classes,
            'eval_metric': 'merror',
            'tree_method': 'hist'  # Use CPU for training
        }

        # Cross-validation parameters
        cv_params = {
            'params': param,
            'dtrain': train,
            'num_boost_round': 20,
            'nfold': 10,  # 10-fold cross-validation
            'metrics': {'merror'},
            'early_stopping_rounds': 10
        }

        # Perform cross-validation
        cv_results = xgb.cv(**cv_params)
        print(cv_results)

        # Determine the best number of boosting rounds
        best_num_boost_round = cv_results.shape[0]

        # Train the model with the best number of boosting rounds
        self.model = xgb.train(param, train, num_boost_round=best_num_boost_round)

    def predict(self, X_test):
        test = xgb.DMatrix(X_test)
        predictions = self.model.predict(test)
        return predictions

def train_xgboost_model(X_train, y_train, num_classes, input_dim):
    xgb_model = XGBoostModel(num_classes, input_dim)
    xgb_model.train(X_train, y_train)
    return xgb_model

def test_xgboost_model(xgb_model, X_test, y_test):
    predictions = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Example usage:
# X_train, X_test, y_train, y_test = ... (load your data here)
# num_classes = ... (number of classes in your dataset)
# input_dim = ... (number of features in your dataset)
# xgb_model = train_xgboost_model(X_train, y_train, num_classes, input_dim)
# accuracy, report = test_xgboost_model(xgb_model, X_test, y_test)
# print(f'Accuracy: {accuracy}')
# print('Classification Report:')
# print(report)
