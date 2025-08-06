## 特徴量選択
最初に選択した特徴量は以下。
- 基本的にplanner_data_とinput_path_msgsの内容とした。
- no_ground_pointcloud, predicted_objects, occupancy_grid に関する変数は、(dummy car等を手動で置かない限り)planning simulationでは変化しないため除外した。
- current_odometry_poseのような2次元/3次元ベクトル値については、L2ノルムと(x,y)についての偏角を特徴量とした。
- 
```
input_path_msg_point_num
input_path_msg_point_position_variance
input_path_msg_point_orientation_mean_direction
current_odometry_pose_L2norm
current_odometry_pose_arg
current_velocity_twist_linear_L2norm
current_velocity_twist_linear_arg
current_velocity_twist_angular_L2norm
current_velocity_twist_angular_arg
velocity_buffer_old_size
velocity_buffer_new_size
velocity_buffer_new_twist_linear_0_L2norm
velocity_buffer_new_twist_linear_0_arg
velocity_buffer_new_twist_angular_0_L2norm
velocity_buffer_new_twist_angular_0_arg
map_data_size
current_acceleration_accel_linear_L2norm
current_acceleration_accel_linear_arg
current_acceleration_accel_angular_L2norm
current_acceleration_accel_angular_arg
input_path_msg_lane_direction_changes
input_path_msg_lane_regulatory_traffic_light
input_path_msg_lane_regulatory_right_of_way
input_path_msg_lane_regulatory_no_stopping_area
input_path_msg_lane_regulatory_road_marking
input_path_msg_lane_regulatory_traffic_sign
```

このときregression_analysis_xgでのテストデータでの結果は、

```
R^2 score: 0.19485503435134888
RMSE: 8628.298351425921
```

また、input_path_msg_point_position_variance, input_path_msg_point_orientation_mean_direction を取り除くと、決定係数が上昇した。position,orientationのvarianceについては、lane_direction_changesによって代替可能と考えたため、取り除いた。

```
R^2 score: 0.504785418510437
RMSE: 6766.822919871217
```

さらに、特徴量を以下に限定したところ、更に決定係数が上昇した。現在の車の動きに関する情報と、input pathのlaneに関する情報のみ残した形になる。

```
current_odometry_pose_L2norm
current_odometry_pose_arg
current_velocity_twist_linear_L2norm
current_velocity_twist_linear_arg
current_velocity_twist_angular_L2norm
current_velocity_twist_angular_arg
current_acceleration_accel_linear_L2norm
current_acceleration_accel_linear_arg
current_acceleration_accel_angular_L2norm
current_acceleration_accel_angular_arg
input_path_msg_lane_direction_changes
input_path_msg_lane_regulatory_traffic_light
input_path_msg_lane_regulatory_right_of_way
input_path_msg_lane_regulatory_no_stopping_area
input_path_msg_lane_regulatory_road_marking
input_path_msg_lane_regulatory_traffic_sign
```

```
R^2 score: 0.7471065521240234
RMSE: 4835.6730363431825
```

さらに、特徴量をinput_path_msg_lane_*のみとした場合も比較的高い決定係数となった。

```
R^2 score: 0.6729429960250854
RMSE: 5499.204025359159
```

## input_path_msg_point
- input_path_msg_pointについては、点群データから特徴量を抽出することを考える。
- 点群のpositionデータから重心を求め、(x,y,z)それぞれについて重心からの距離の二乗和を求め、そのL2ノルムを"input_path_msg_point_position_variance"とした。
- また、点群のorientationデータから"input_path_msg_point_orientation_mean_direction"として(x,y,z,w)それぞれについて平均を求め、そのL2ノルムを出した。
    - これが大きいほど方角のばらつきが小さいと考えられる。

- lanelet mapを元に、パスの属性データを追加で手に入れた。これを元にvelocity_plannerの各コンポーネントが動いていると考えられる。
    - lane_subtype : 常にroad
    - lane_turn_direction : 曲がる方向
        - direction changeの回数を特徴量とした。
    - lane_regulatory_type
        - traffic_light, right_of_way, traffic_sign, no_stopping_area, road_marking
        - 各タイプの出現回数を特徴量とした。

## 特徴量抽出

make_df.pyの関数calc_xx_feature()において行った。
