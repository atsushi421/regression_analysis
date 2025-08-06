import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
import re
import os


def L2norm(a):
    s = 0
    for i in range(len(a)):
        s += float(a[i]) * float(a[i])
    s = math.sqrt(s)
    return s

def calc_position_feature(point_pos_list):
    # calc centroid
    centroid = [0, 0, 0]
    for pos in point_pos_list:
        centroid[0] += pos[0]
        centroid[1] += pos[1]
        centroid[2] += pos[2]
    centroid[0] /= len(point_pos_list)
    centroid[1] /= len(point_pos_list)
    centroid[2] /= len(point_pos_list)
    # calc variance
    variance = [0, 0, 0]
    for pos in point_pos_list:
        variance[0] += (pos[0] - centroid[0]) ** 2
        variance[1] += (pos[1] - centroid[1]) ** 2
        variance[2] += (pos[2] - centroid[2]) ** 2
    variance[0] /= len(point_pos_list)
    variance[1] /= len(point_pos_list)
    variance[2] /= len(point_pos_list)

    # calc L2 norm of variance
    variance_l2 = math.sqrt(variance[0] ** 2 + variance[1] ** 2 + variance[2] ** 2)
    return variance_l2

def calc_orientation_feature(point_ori_list):
    #calc mean direction
    mean_direction = [0, 0, 0, 0]
    for ori in point_ori_list:
        mean_direction[0] += ori[0]
        mean_direction[1] += ori[1]
        mean_direction[2] += ori[2]
        mean_direction[3] += ori[3]
    mean_direction[0] /= len(point_ori_list)
    mean_direction[1] /= len(point_ori_list)
    mean_direction[2] /= len(point_ori_list)
    mean_direction[3] /= len(point_ori_list)
    # calc L2 norm of mean direction
    mean_direction_l2 = math.sqrt(mean_direction[0] ** 2 + mean_direction[1] ** 2 + mean_direction[2] ** 2 + mean_direction[3] ** 2)
    return mean_direction_l2

def calc_direction_feature(direction_list):
    # direction_list : list of string (e.g. ['left', 'right', 'straight'])
    # count the number of direction changes
    result = 0
    for i in range(len(direction_list)):
        if i>0 and direction_list[i] != direction_list[i-1]:
            result += 1
    return result

def calc_regulartory_feature(regulartory_list, type_str):
    # regulartory_list : list of string (e.g. ['stopline', 'traffic_light'])
    # count the number of regulartory changes to type_str
    result = 0
    for i in range(len(regulartory_list)):
        if type_str in regulartory_list[i]:
            if i > 0 and type_str not in regulartory_list[i-1]:
                result += 1
    return result

def make_df(time_log, variables_log, variables_log_string):
    df = pd.DataFrame()
    logdata = {}
    part_idx_mx = 0
    turnaround_times = {}
    with open(time_log, mode="r", encoding="utf-8") as f:
        for line in f:
            ret = line.rstrip().split()
            session_name, part_idx, loop_idx, timestamp, data = ret
            part_idx = int(part_idx)
            loop_idx = int(loop_idx)
            timestamp = int(timestamp)

            if not part_idx in logdata:
                logdata[part_idx] = []
                part_idx_mx = max(part_idx_mx, part_idx)

            logdata[part_idx].append(timestamp)
    for i in range(0, part_idx_mx):
        turnaround_times[i] = []
        for j in range(len(logdata[0])):
            turnaround_times[i].append(logdata[i+1][j] - logdata[i][j])

    df['time'] = turnaround_times[1]
    var_id = dict()
    var_names = []
    vi = 0
    pair_vec = []
    is_single = []
    point_pos_list = []
    point_ori_list = []
    before_loop_idx = 0
    with open(variables_log, mode="r", encoding="utf-8") as f:
        for line in f:
            ret = line.rstrip().split()
            session_name = ret[0]
            loop_idx = int(ret[1])
            if len(turnaround_times[1])<=loop_idx:
                continue
            #if loop_idx != before_loop_idx:
                # process point data
                # if len(point_pos_list) > 0:
                #     var_name = "input_path_msg_point_position_variance"
                #     if not (var_name in var_id.keys()):
                #         var_id[var_name] = vi
                #         vi += 1
                #         var_names.append(var_name)
                #         pair_vec.append([])
                #     pair_vec[var_id[var_name]].append((calc_position_feature(point_pos_list), before_loop_idx))
                #     point_pos_list = []
                # if len(point_ori_list) > 0:
                #     var_name = "input_path_msg_point_orientation_mean_direction"
                #     if not (var_name in var_id.keys()):
                #         var_id[var_name] = vi
                #         vi += 1
                #         var_names.append(var_name)
                #         pair_vec.append([])
                #     pair_vec[var_id[var_name]].append((calc_orientation_feature(point_ori_list), before_loop_idx))
                #     point_ori_list = []
                # before_loop_idx = loop_idx
            var_name = ret[2]
            # restrict variables
            if not re.match(r"^current.*",var_name) :
                continue
            if re.match(r"^velocity_buffer.*", var_name):
                continue
            if re.match(r"^input_path_msg_point_position.*", var_name):
                point_pos_list.append((float(ret[3]), float(ret[4]), float(ret[5])))
                continue
            if re.match(r"^input_path_msg_point_orientation.*", var_name):
                point_ori_list.append((float(ret[3]), float(ret[4]), float(ret[5]), float(ret[6])))
                continue
            if re.match(r"^input_path_msg.*",var_name) and var_name != "input_path_msg_point_num":
                continue
            if re.match(r"^occupancy_grid.*", var_name):
                continue
            if re.match(r"^no_ground_pointcloud.*", var_name):
                continue
            if re.match(r"^predicted_objects.*", var_name):
                continue
            if re.match(r"^map_data_size", var_name):
                continue
            var_name_l2 = var_name + "_L2norm"
            var_name_arg = var_name + "_arg"
            data = 0
            #print(len(ret))
            if len(ret) < 4:
                continue
            # calculate L2 norm / arg
            
            if len(ret) == 4:
                is_single.append(1)
                if not (var_name in var_id.keys()):
                    var_id[var_name] = vi
                    var_names.append(var_name)
                    vi+=1
                    pair_vec.append([])
            else:
                is_single.append(0)
                if not (var_name_l2 in var_id.keys()):
                    var_id[var_name_l2] = vi
                    vi += 1
                    var_id[var_name_arg] = vi
                    vi += 1
                    var_names.append(var_name_l2)
                    var_names.append(var_name_arg)
                    pair_vec.append([])
                    pair_vec.append([])
                
            if len(ret) == 4:
                pair_vec[var_id[var_name]].append((float(ret[3]),loop_idx))
            else:
                pair_vec[var_id[var_name_l2]].append((L2norm(ret[3:]), loop_idx))
                pair_vec[var_id[var_name_arg]].append((math.atan2(float(ret[4]),float(ret[3])),loop_idx))
    
    direction_list = []
    regulartory_list = []
    before_loop_idx = 0
    with open(variables_log_string, mode="r", encoding="utf-8") as f:
        for line in f:
            ret = line.rstrip().split()
            session_name = ret[0]
            loop_idx = int(ret[1])
            if len(turnaround_times[1]) <= loop_idx:
                continue
            var_name = ret[2]
            if loop_idx != before_loop_idx: 
                if len(direction_list) > 0:
                    var_name = "input_path_msg_lane_direction_changes"
                    if not (var_name in var_id.keys()):
                        var_id[var_name] = vi
                        vi += 1
                        var_names.append(var_name)
                        pair_vec.append([])
                    pair_vec[var_id[var_name]].append((calc_direction_feature(direction_list), before_loop_idx))
                    direction_list = []
                if len(regulartory_list) > 0:
                    regulartory_types = ["traffic_light", "right_of_way", "no_stopping_area", "road_marking", "traffic_sign"]
                    for regulartory_type in regulartory_types:
                        var_name = "input_path_msg_lane_regulatory_" + regulartory_type
                        if not (var_name in var_id.keys()):
                            var_id[var_name] = vi
                            vi += 1
                            var_names.append(var_name)
                            pair_vec.append([])
                        pair_vec[var_id[var_name]].append((calc_regulartory_feature(regulartory_list, regulartory_type), before_loop_idx))
                    regulartory_list = []
            if re.match(r"^input_path_msg_lane_regulatory_type_.*", var_name):
                if len(ret) >= 4:
                    l = []
                    for i in range(3, len(ret)):
                        l.append(ret[i])
                    regulartory_list.append(l)
                else: 
                    regulartory_list.append([])
            elif re.match(r"^input_path_msg_lane_turn_direction_.*", var_name):
                if len(ret) >= 4:
                    for i in range(3,len(ret)):
                        direction_list.append(ret[i])
                else:
                    direction_list.append("unknown")
            
            before_loop_idx = loop_idx

    for i in range(len(pair_vec)):
        lst = [0 for _ in range(len(turnaround_times[1]))]
        for p in pair_vec[i]:
            lst[p[1]] = p[0]
        print(var_names[i])
        df[var_names[i]] = lst
    
    return df

def make_df_in_directory(directory):
# for all file, calculate df by (elapsed_time_log_xxx_0, variables_log_xxx_0) and concat them
    df_list = []
    for filename in os.listdir(directory):
        if filename.startswith("elapsed_time_log_"):
            time_log = os.path.join(directory, filename)
            variables_log = time_log.replace("elapsed_time_log_", "variables_log_")
            variables_log_string = variables_log + "_string"
            if not os.path.exists(variables_log) or not os.path.exists(variables_log_string):
                continue
            df = make_df(time_log, variables_log, variables_log_string)
            # save df to csv
            # csv_filename = os.path.join(directory, filename.replace("elapsed_time_log_", "df_")+".csv")
            # df.to_csv(csv_filename, index=False)
            # print(f"Processed {filename} and saved to {csv_filename}")
            # append df to list
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

if __name__ == "__main__":
    # python elapsed_time_parser.py time_log variables_log variables_log_string
    df = make_df(sys.argv[1], sys.argv[2], sys.argv[3])
    # print df
    print(df.head())
    df.to_csv("output.csv", index=False)


