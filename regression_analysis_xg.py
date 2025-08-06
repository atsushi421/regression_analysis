import make_df
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import xgboost as xgb

# regression analysis by xgboost
def regression_analysis(train_df, test_df):
    # Assuming 'time' is the dependent variable and the rest are independent variables
    X = train_df.drop(columns=['time'])
    y = train_df['time']
    feature_names = X.columns.tolist()
    X_test = test_df.drop(columns=['time'])
    y_test = test_df['time']
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )  

    # use scaler to normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns=feature_names)
    X_test = scaler.transform(X_test)
    # Train the model
    xgb_reg.fit(X, y)
    # Print feature importances
    print("Feature importances:", xgb_reg.feature_importances_)
    # Plotting feature importances
    plt.figure(figsize=(20, 6))
    xgb.plot_importance(xgb_reg, importance_type='weight',ax=plt.gca())
    plt.title('Feature Importances')
    plt.savefig('feature_importances_xg.png')  
    # Predicting on the test set
    y_pred = xgb_reg.predict(X_test)
    # calc R^2 and RMSE
    r2_score = xgb_reg.score(X_test, y_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"R^2 score: {r2_score}")
    print(f"RMSE: {rmse}")
    # Plotting actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Time')
    plt.ylabel('Predicted Time')
    plt.title('Actual vs Predicted Time')
    plt.savefig('actual_vs_predicted_xg.png')
    
def regression_analysis_cross_validation(all_df, test_df):
    # Assuming 'time' is the dependent variable and the rest are independent variables
    X = all_df.drop(columns=['time'])
    y = all_df['time']
    feature_names = X.columns.tolist()
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )  
    # use scaler to normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns=feature_names)
    
    # Create a DMatrix for cross-validation
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    
    # Perform cross-validation
    cv_results = xgb.cv(
        params=xgb_reg.get_xgb_params(),
        dtrain=dtrain,
        num_boost_round=300,
        nfold=5,
        metrics='rmse',
        early_stopping_rounds=10,
        verbose_eval=True
    )
    
    print("Cross-validation results:", cv_results)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=cv_results.shape[0],
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X, y)
    # test the model on test set
    X_test = test_df.drop(columns=['time'])
    y_test = test_df['time']
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    y_pred = model.predict(X_test)
    # calc R^2 and RMSE
    r2_score = model.score(X_test, y_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"R^2 score: {r2_score}")
    print(f"RMSE: {rmse}")
    # Plotting actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Time')
    plt.ylabel('Predicted Time')
    plt.title('Actual vs Predicted Time')
    plt.savefig('actual_vs_predicted_xg_cv.png')
    # Plotting feature importances
    plt.figure(figsize=(20, 6))
    xgb.plot_importance(model, importance_type='weight', ax=plt.gca())
    plt.title('Feature Importances')
    plt.savefig('feature_importances_xg_cv.png')


if __name__ == "__main__":
    # python regression_analysis.py

    train_df = make_df.make_df_in_directory("./train_log")
    test_df = make_df.make_df_in_directory("./test_log")
    # Perform regression analysis
    #regression_analysis(train_df,test_df)
    regression_analysis_cross_validation(train_df, test_df)