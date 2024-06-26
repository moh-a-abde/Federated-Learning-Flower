import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple

class Net:
    def __init__(self, num_classes: int, input_dim: int) -> None:
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='reg:squarederror' if num_classes == 1 else 'multi:softprob',
            num_class=num_classes if num_classes > 1 else None,
            random_state=42
        )
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

def train(model: XGBoostModel, X_train: np.ndarray, y_train: np.ndarray, epochs: int, device: str = 'cpu'):
    # XGBoost doesn't use GPU by default, so we ignore the device parameter
    # We'll use the number of epochs as n_estimators
    model.model.n_estimators = epochs
    model.model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_train, y_train)], 
        early_stopping_rounds=10,
        verbose=True
    )

def test(model: XGBoostModel, X_test: np.ndarray, y_test: np.ndarray, device: str = 'cpu') -> Tuple[float, float]:
    # Again, we ignore the device parameter for XGBoost
    y_pred = model.forward(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = model.model.score(X_test, y_test)
    return mse, accuracy

def early_stopping(val_losses, patience=10):
    if len(val_losses) > patience:
        if all(val_losses[-1] > val_losses[-(i+2)] for i in range(patience)):
            return True
    return False
