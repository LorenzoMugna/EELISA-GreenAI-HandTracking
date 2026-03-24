import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 1. TODO import your dataset and preprocess it to get features (X) and labels (y)
# For example, if you have a CSV file, you can use pandas to load it:
df = pd.read_csv('data/parsed_data.csv', sep=';')
print ("Dataset loaded successfully. Sample data:")
print(df.head())
X = df.drop(columns=['label'])  
y = df['label']

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)


# 3. Initialize the XGBoost Classifier
# Key parameters for multi-class classification are 'objective' and 'num_class'
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', # Use multi:softmax for multiple classes
    num_class=2,               # The number of classes in your dataset
    max_depth=4,               # Maximum depth of a tree
    learning_rate=0.1,         # Step size shrinkage used to prevent overfitting
    n_estimators=100,          # Number of boosting rounds (trees),
    eval_metric='mlogloss'     # Evaluation metric for multi-class
)

# 4. Cross-validation sul training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb_model,
    X_train,
    y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print("Cross-validation (5-fold) sul training set")
print(f"Accuratezza media CV: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# 5. Train finale sul training set completo
print("\nTraining finale sul training set completo...")
xgb_model.fit(X_train, y_train, verbose=False)

# 6. Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))