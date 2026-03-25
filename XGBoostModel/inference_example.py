import xgboost as xgb
import numpy as np
import joblib

# 1. Initialize an empty Booster
loaded_model = xgb.Booster()

# 2. Load the saved weights and configuration
loaded_model.load_model("PredictionModelNoSpike.json")

# 2b. Load the scaler used during training
scaler = joblib.load("scalerNoSpike.pkl")

# 3. Make predictions (Assuming X_new_gpu is a CuPy array of new data)
# Note: Native XGBoost expects a DMatrix for prediction

#labels: ['palm_normal_y', 'digit_0_distance', 'digit_1_distance', 'digit_2_distance', 'digit_3_distance', 'digit_4_distance']
Data = [0,0,0,0,0,0]  # Replace with your actual data for prediction (must be in the same order as training features)
Data = np.array(Data, dtype=np.float32).reshape(1, -1)
Data = scaler.transform(Data)
dnew = xgb.DMatrix(Data)

# 4. Predict the class probabilities for the new data
# ouput will be a 2D array where each row corresponds to the predicted probabilities for each class
prediction = loaded_model.predict(dnew)

print(prediction)