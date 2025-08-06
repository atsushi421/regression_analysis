# behavior_velocity_planner_log

## elapsed_time_variables_parser.py
[pmu_analyzer](https://github.com/taisa1/pmu_analyzer)による出力ログを元に実行時間との相関を計算し、プロットを作成します。
データの取得・前処理を行うpart0、publishを行うpart2は除き、plannerによってgeneratePath()を行うpart1の実行時間を考慮しています。

プロットは`var_fig/`以下に保存されます。標準出力に相関係数が昇順で出力されます。実行時間自体のプロットはpdfで出力されます。

以下のように実行します。
```
python3 elapsed_time_variables_parser.py <elapsed time log path> <variables log path>
```

## regression_analysis_{lgb|xg}.py
実行時間を目的変数とした回帰分析のプロトタイプです。lgbはLightGBM、xgはXGBoostを使っています。
train_log/以下のログ（複数でもOK）を用いて学習され、test_log/以下のログを用いてテストされます。
テスト結果として、actual_vs_predicted_{lgb|xg}.pngとfeature_importances_{lgb|xg}.png が出力されます。

xgを用いる場合は、最適なステップ数の計算に交差検証が用いられます。こちらの方が精度が高いはずです。

## make_df.py
上記のスクリプトで用いられる、ログから特徴量を計算し、DataFrame形式で保存するスクリプトです。


## node.cpp
計測に用いた `autoware_core/planning/behavior_velocity_planner/autoware_behavior_velocity_planner/src/node.cpp` です。

## 計測について
behavior_velocity_plannerを含む、behavior_planningのcomponent_containerについてCPUの隔離・周波数固定を行いました。
その上でPlanning Simulationを行い、適当なパスを指定して実行しました。