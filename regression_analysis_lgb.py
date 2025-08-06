import make_df
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score

# regression analysis by lightgbm
def regression_analysis(train_df, test_df):
    # Assuming 'time' is the dependent variable and the rest are independent variables
    X = train_df.drop(columns=['time'])
    y = train_df['time']
    feature_names = X.columns.tolist()
    # Use scaler to normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    # Create a LightGBM dataset
    lgb_train = lgb.Dataset(X, y) 

    # Set parameters for LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    # Train the model
    model = lgb.train(params, lgb_train, num_boost_round=100)

    # Print feature importances
    print("Feature importances:", model.feature_importance())

    # Plotting feature importances
    lgb.plot_importance(model, max_num_features=10, figsize=(20, 6))
    plt.title('Feature Importances')
    plt.savefig('feature_importances_lgb.png')

    # Predicting on the test set
    X_test = test_df.drop(columns=['time'])
    # Use the same scaler to transform the test set
    X_test = scaler.transform(X_test)
    y_test = test_df['time']
    y_pred = model.predict(X_test)
   
    # calc R^2 and RMSE
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"R^2 score: {r2}")
    print(f"RMSE: {rmse}")
    # Plotting actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Time')
    plt.ylabel('Predicted Time')
    plt.title('Actual vs Predicted Time')
    plt.savefig('actual_vs_predicted_lgb.png')
    

if __name__ == "__main__":
    # python regression_analysis.py

    train_df = make_df.make_df_in_directory("./train_log")
    test_df = make_df.make_df_in_directory("./test_log")
    # Perform regression analysis
    regression_analysis(train_df,test_df)