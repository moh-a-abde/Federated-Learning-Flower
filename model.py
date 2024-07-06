import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

def train_xgboost(data):
    # Encode categorical features
    label_encoder = LabelEncoder()
    # Print unique values in the 'label' column
    unique_labels = data['label'].unique()
    print("Unique values in 'label' column:", unique_labels)
    data['label'] = label_encoder.fit_transform(data['label'])

    # Separate the 'ts' column into a different dataset
    ts_data = data[['ts']]

    # Select features and target
    X = data.drop(columns=['label', 'ts', 'uid'])
    y = data['label']

    # Convert categorical features to numerical
    X = pd.get_dummies(X, columns=['id.orig_h', 'id.resp_h', 'proto', 'conn_state', 'history'])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set the test size
    tsz = 0.30

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=tsz, stratify=y, random_state=42)
    l = len(set(y))
    startTrainTime = time.time()
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
    TrainTime = (time.time() - startTrainTime)
    startTestTime = time.time()
    predictions = model.predict(train)
    TestTime = (time.time() - startTestTime)
    print(f"Training Time: {TrainTime} seconds")
    print(f"Testing Time: {TestTime} seconds")
    accuracy = accuracy_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    print('XGBoost Model Training Metrics:')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    return model
