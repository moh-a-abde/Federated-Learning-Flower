import pandas as pd
import xgboost as xgb
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

def train_xgboost(trainloader: DataLoader):
    data_list = [(features.numpy(), labels.numpy()) for features, labels in trainloader]

    X = np.concatenate([x[0] for x in data_list])
    y = np.concatenate([x[1] for x in data_list])

    dtrain = xgb.DMatrix(X, label=y)

    param = {
        'max_depth': 6,
        'eta': 0.35,
        'objective': 'multi:softmax',
        'num_class': len(set(y)),
        'eval_metric': 'merror',
        'tree_method': 'hist'
    }

    cv_results = xgb.cv(param, dtrain, num_boost_round=20, nfold=10, early_stopping_rounds=10)
    best_num_boost_round = cv_results.shape[0]
    model = xgb.train(param, dtrain, num_boost_round=best_num_boost_round)

    predictions = model.predict(dtrain).astype(int)
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)

    print('XGBoost Model Training Metrics:')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    return model
