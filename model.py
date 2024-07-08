import pandas as pd
import xgboost as xgb
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_xgboost(trainloader: DataLoader):
    # Combine data from DataLoader into a DataFrame
    data_list = []
    for features, labels in trainloader:
        features_np = features.numpy()
        labels_np = labels.numpy()
        data_list.append((features_np, labels_np))

    X = np.concatenate([x[0] for x in data_list])
    y = np.concatenate([x[1] for x in data_list])

    # Set the test size
    tsz = 0.30

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsz, stratify=y, random_state=42)
    
    l = len(set(y))
    
    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)
    
    param = {
        'max_depth': 6,
        'eta': 0.35,
        'objective': 'multi:softmax',
        'num_class': l,
        'eval_metric': 'merror',
        'tree_method': 'hist'
    }
    cv_params = {
        'params': param,
        'dtrain': train,
        'num_boost_round': 20,
        'nfold': 10,
        'metrics': {'merror'},
        'early_stopping_rounds': 10
    }
    cv_results = xgb.cv(**cv_params)
    print(cv_results)
    best_num_boost_round = cv_results.shape[0]
    model = xgb.train(param, train, num_boost_round=best_num_boost_round)

    predictions = model.predict(test)
    predictions = predictions.astype(int)  # Ensure predictions are integer type

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print('XGBoost Model Training Metrics:')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    
    return model
