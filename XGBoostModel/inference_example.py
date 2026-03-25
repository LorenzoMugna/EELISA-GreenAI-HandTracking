import xgboost as xgb
import numpy as np
import joblib

# 1. Initialize an empty Booster
loaded_model = xgb.Booster()

# 2. Load the saved weights and configuration
loaded_model.load_model("PredictionModelWithSpike.json")

# 2b. Load the scaler used during training
scaler = joblib.load("scalerWithSpike.pkl")

# 3. Make predictions (Assuming X_new_gpu is a CuPy array of new data)
# Note: Native XGBoost expects a DMatrix for prediction

# the channels are from 0 to 4 the fingers and 5 is the palm y normal, the features are the same as in the training phase, but you can replace them with your actual data for prediction
#labels: ['ch0_rate_hz', 'ch1_rate_hz', 'ch2_rate_hz', 'ch3_rate_hz', 'ch4_rate_hz', 'ch0_var_isi_ms', 'ch1_var_isi_ms', 'ch2_var_isi_ms', 'ch3_var_isi_ms', 'ch4_var_isi_ms']
Data = [0,0,0,0,0,0,0,0,0,0]  # Replace with your actual data for prediction (must be in the same order as training features)
Data = np.array(Data, dtype=np.float32).reshape(1, -1)
Data = scaler.transform(Data)
dnew = xgb.DMatrix(Data)

# 4. Predict the class probabilities for the new data
# ouput will be a 2D array where each row corresponds to the predicted probabilities for each class
prediction = loaded_model.predict(dnew)

print(prediction)