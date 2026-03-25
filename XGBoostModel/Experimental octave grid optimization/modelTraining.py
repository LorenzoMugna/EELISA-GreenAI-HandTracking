import xgboost as xgb
import cupy as cp
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ... [Load your data into X and y as before] ...
df = pd.read_csv('../data/parsed_spikes_data_total.csv')
X = df[['ch0_rate_hz', 'ch1_rate_hz', 'ch2_rate_hz', 'ch3_rate_hz', 'ch4_rate_hz', 'ch5_rate_hz', 'ch0_var_isi_ms', 'ch1_var_isi_ms', 'ch2_var_isi_ms', 'ch3_var_isi_ms', 'ch4_var_isi_ms', 'ch5_var_isi_ms']]
y = df['label']

# Normalize the dataset
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X = pd.DataFrame(X_normalized, columns=X.columns)

# Export the scaler for future use
joblib.dump(scaler, 'scaler.pkl')
print(f"Test solo distanze scalari - Dataset normalized")
print(f"Scaler exported to 'scaler.pkl'")


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    stratify=y,
    random_state=42
)

# Validation split interno solo per Optuna (evita leakage sul test)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=42
)

# 1. Convert to CuPy arrays
X_train_gpu = cp.asarray(X_train.to_numpy())
y_train_gpu = cp.asarray(y_train.to_numpy())
X_valid_gpu = cp.asarray(X_valid.to_numpy())
y_valid_gpu = cp.asarray(y_valid.to_numpy())
X_test_gpu = cp.asarray(X_test.to_numpy())
y_test_gpu = cp.asarray(y_test.to_numpy())

# 2. Use QuantileDMatrix (Fastest memory structure for GPU 'hist' method)
dtrain = xgb.QuantileDMatrix(X_train_gpu, label=y_train_gpu)
dvalid = xgb.QuantileDMatrix(X_valid_gpu, label=y_valid_gpu, ref=dtrain)

def objective(trial):
    # 3. Define the search space dynamically
    param = {
        'objective': 'multi:softprob',
        'num_class': y.nunique(),
        'device': 'cuda:0',
        'tree_method': 'hist',
        'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'eval_metric': 'mlogloss'
    }
    
    # 4. Use Optuna's Pruning Callback
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-mlogloss')
    
    # 5. Native XGBoost training (bypasses Scikit-Learn entirely)
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=trial.suggest_int('n_estimators', 100, 800),
        evals=[(dvalid, 'valid')],
        callbacks=[pruning_callback],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    
    # Minimizziamo mlogloss su validation
    return bst.best_score

# 6. Run the optimization study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

print(f"Best parameters: {study.best_params}")

# 7. Train finale sul training completo (train_full) con i best params
best_params = {
    'objective': 'multi:softprob',
    'num_class': y.nunique(),
    'device': 'cuda:0',
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'seed': 42,
    **{k: v for k, v in study.best_params.items() if k != 'n_estimators'}
}

X_train_full_gpu = cp.asarray(X_train_full.to_numpy())
y_train_full_gpu = cp.asarray(y_train_full.to_numpy())
dtrain_full = xgb.QuantileDMatrix(X_train_full_gpu, label=y_train_full_gpu)
dtest = xgb.QuantileDMatrix(X_test_gpu, label=y_test_gpu, ref=dtrain_full)

final_bst = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=study.best_params['n_estimators'],
    verbose_eval=False
)

# 8. Evaluate on hold-out test
y_proba = final_bst.predict(dtest)
y_proba_np = cp.asnumpy(y_proba) if isinstance(y_proba, cp.ndarray) else y_proba
y_pred = np.argmax(y_proba_np, axis=1)
y_test_np = cp.asnumpy(y_test_gpu)

accuracy = accuracy_score(y_test_np, y_pred)
print(f"Final Accuracy (hold-out test): {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred))

# 9. Feature importance plot (gain)
importance_dict = final_bst.get_score(importance_type='gain')

importance_series = pd.Series({
    X.columns[int(feature_idx[1:])]: importance_value
    for feature_idx, importance_value in importance_dict.items()
}).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importance_series.plot(kind='barh')
plt.title('XGBoost Feature Importance (gain)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()