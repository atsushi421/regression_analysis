# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains performance analysis tools for Autoware's behavior velocity planner component. It analyzes execution time correlations with various planner variables and performs regression analysis using machine learning models to predict planner performance.

## Core Components

### Data Processing Pipeline
- **make_df.py**: Core data processing module that converts raw log files into feature-engineered DataFrames
  - Extracts features from position/orientation data using L2 norms and variance calculations
  - Processes lane information from lanelet maps (traffic lights, right of way, road markings, etc.)
  - Normalizes vehicle state data (odometry, velocity, acceleration)

### Analysis Scripts
- **elapsed_time_variables_parser.py**: Correlation analysis between execution time and planner variables
  - Usage: `python3 elapsed_time_variables_parser.py <elapsed_time_log_path> <variables_log_path>`
  - Outputs correlation plots to `var_fig/` directory and correlation coefficients to stdout
  - Focuses on part1 execution time (generatePath() function)

- **regression_analysis_xg.py**: XGBoost regression model for execution time prediction
  - Uses cross-validation for optimal hyperparameter tuning
  - Trains on `train_log/` data, tests on `test_log/` data
  - Outputs: `actual_vs_predicted_xg.png` and `feature_importances_xg.png`

- **regression_analysis_lgb.py**: LightGBM regression model alternative
  - Similar functionality to XGBoost version but with LightGBM backend
  - Outputs: `actual_vs_predicted_lgb.png` and `feature_importances_lgb.png`

### Data Structure
- **train_log/**: Training data logs (elapsed_time_log_* and variables_log_* files)
- **test_log/**: Test data logs for model validation
- **sample_log/**: Sample log files for testing
- **var_fig/**: Output directory for correlation plots

## Development Commands

### Running Analysis
```bash
# Correlation analysis
python3 elapsed_time_variables_parser.py train_log/elapsed_time_log_28441_0 train_log/variables_log_28441_0

# XGBoost regression (recommended for higher accuracy)
python3 regression_analysis_xg.py

# LightGBM regression
python3 regression_analysis_lgb.py
```

### Dependencies
The scripts require the following Python packages:
- pandas, numpy
- matplotlib
- scikit-learn
- xgboost, lightgbm
- math, sys, re, os (standard library)

## Feature Engineering

### Selected Features (Current Best Performance)
Based on feature selection analysis documented in `feature_select.md`, the optimal feature set includes:
- Vehicle state: current_odometry_pose, current_velocity_twist, current_acceleration_accel (L2 norms and angular components)
- Lane information: input_path_msg_lane_regulatory_* (traffic_light, right_of_way, no_stopping_area, road_marking, traffic_sign)
- Path characteristics: input_path_msg_lane_direction_changes

### Feature Extraction Process
- 2D/3D vector values converted to L2 norms and angular components
- Point cloud data processed for position variance and orientation consistency
- Lane regulatory information extracted from lanelet maps
- Vehicle dynamics normalized using StandardScaler

## Performance Metrics

Current best model (XGBoost with selected features):
- R² score: 0.7471
- RMSE: 4835.67 microseconds

## Key Architecture Notes

- The analysis focuses on Autoware's behavior_velocity_planner component performance
- Log data comes from pmu_analyzer output during Planning Simulation
- CPU isolation and frequency fixing were applied during measurement for consistent results
- Only part1 (generatePath()) execution time is analyzed, excluding data acquisition (part0) and publishing (part2)
