import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

def preprocess_data(df):
    data = df.copy()
    
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()

    # Encode categorical variables
    for col in ['sex', 'smoker', 'region']:
        data[f'{col}_enc'] = LabelEncoder().fit_transform(data[col])

    features = ['age', 'sex_enc', 'bmi', 'children', 'smoker_enc', 'region_enc']
    X = data[features]
    y = data['charges']
    
    return X, y

def train_insurance_model():
    mlflow.set_experiment("insurance_cost_prediction")

    data = pd.read_csv("data/insurance.csv") 
    print(f"Dataset shape: {data.shape}")
    print(f"Target variable stats:\n{data['charges'].describe()}")
    
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Def hyperparameters
    n_estimators = 100
    max_depth = 15
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Train
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        # Test
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2_score", train_r2)

        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2_score", test_r2)
        
        # Log feature importances
        feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        

        mlflow.sklearn.log_model(model, "random_forest_model")
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/random_forest_model"
        mlflow.register_model(model_uri, "random_forest_model")
        
        print(f"Train MSE: {train_mse:.2f}$, RMSE: {train_rmse:.2f}$, MAE: {train_mae:.2f}$, R2: {train_r2:.4f}")
        print(f"Test MSE: {test_mse:.2f}$, RMSE: {test_rmse:.2f}$, MAE: {test_mae:.2f}$, R2: {test_r2:.4f}")
        print("Feature Importances:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        return model, test_mse, test_rmse, test_r2

if __name__ == "__main__":
    train_insurance_model()