import xgboost as xgb
import pandas as pd
import cupy as cp
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 1. TODO import your dataset and preprocess it to get features (X) and labels (y)
# For example, if you have a CSV file, you can use pandas to load it:
df = pd.read_csv('data/parsed_data.csv', sep=';')
X = df[['digit_0_distance', 'digit_1_distance', 'digit_2_distance', 'digit_3_distance', 'digit_4_distance']]  
y = df['label']
print(f"Test solo distanze scalari")

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    stratify=y,
    random_state=42
)


# 3. Initialize the XGBoost Classifier
# Key parameters for multi-class classification are 'objective' and 'num_class'
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob', # Use multi:softmax for multiple classes
    num_class=y.nunique(),               # The number of classes in your dataset
    device='cuda:0',
    tree_method='hist',
    n_estimators=140,
    min_child_weight=2, 
    max_depth=13, 
    learning_rate=0.1
)

# # Optional: You can perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
# param_grid = {
#     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
#     'max_depth': range(3, 15),
#     'min_child_weight': range(1, 10),
#     'n_estimators': range(50, 200, 10)
# }

# grid_search = RandomizedSearchCV(
#     estimator=xgb_model, 
#     param_distributions=param_grid,
#     n_iter = 100, # Number of parameter settings that are sampled
#     scoring='roc_auc_ovr', # Use ROC-AUC for multi-class classification 
#     n_jobs=1, # Use all available cores
#     cv=5,      # 5-fold cross-validation
#     verbose=3
# )

X_train_gpu = cp.asarray(X_train.to_numpy())
X_test_gpu = cp.asarray(X_test.to_numpy())

# grid_search.fit(X_train_gpu, y_train)
    
# 5. Print the best parameters and the best score
# print(f"Best parameters found: {grid_search.best_params_}")
# print(f"Best ROC-AUC score: {grid_search.best_score_}")

# optimized_xgb = grid_search.best_estimator_

# 4. Cross-validation sul training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb_model,
    X_train_gpu,
    y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=1
)

print("Cross-validation (5-fold) sul training set")
print(f"Accuratezza media CV: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# 5. Train finale sul training set completo
print("\nTraining finale sul training set completo...")
# optimized_xgb = xgb.XGBClassifier(
#     objective='multi:softprob',
#     num_class=y.nunique(),
#     device='cuda',
#     tree_method='hist',
#     **grid_search.best_estimator_.get_params()
# )

xgb_model.fit(X_train_gpu, y_train, verbose=False)

# 6. Make predictions on the test set
y_pred = xgb_model.predict(X_test_gpu)
y_pred = cp.asnumpy(y_pred)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))