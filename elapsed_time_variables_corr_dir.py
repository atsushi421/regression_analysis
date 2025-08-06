# usage: python3 elapsed_time_variables_parser.py <elapsed time log path> <variables log path>

import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import pandas as pd
import re
import make_df

logdata = {}

def L2norm(a):
    s = 0
    for i in range(len(a)):
        s += float(a[i]) * float(a[i])
    s = math.sqrt(s)
    return s

part_idx_mx = 0
turnaround_times = {}

def parse_logs(directory_path):
    global logdata, part_idx_mx, turnaround_times

    df = make_df.make_df_in_directory(directory_path)
    turnaround_times[1] = df['time'].tolist()

    # calc correlation
    corr_idx = []
    for i in range(len(df.columns)): 
        if df.columns[i] == 'time':
            continue
        corr = df.iloc[:, i].corr(df['time'])
        if not math.isnan(corr):
            corr_idx.append((corr,i))
    # sort by corr
    corr_idx = sorted(corr_idx)

    # output
    for id in range(len(corr_idx)):
        corr = corr_idx[id][0]
        i = corr_idx[id][1]
        var_name = df.columns[i]
        print(var_name+" corr = {}".format(corr))
        plt.figure(figsize=(8, 8))
        plt.title(var_name+" corr={}".format(corr))
        plt.xlabel(var_name + " L2 norm")
        plt.ylabel("turn-around time (us) in part 1")
        l1 = df.iloc[:, i].tolist()
        l2 = df['time'].tolist()
        plt.scatter(l1, l2)
        plt.savefig("var_fig/{}.png".format(var_name))

def visualize(session_name, bins=50):
    for i in range(1,1):
        fig = plt.figure(figsize=(16, 16))
        max_value = max(turnaround_times[i])

        ax0 = fig.add_subplot(2, 1, 1)
        ax0.set_title("{}: part {} - elapsed time time-series".format(session_name, i))
        ax0.set_xlabel("sample index")
        ax0.set_ylabel("turn-around time (us)")
        ax0.set_ylim([0, max_value])
        ax0.plot(turnaround_times[i])

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.set_title("{}: part {} - elapsed time histgram".format(session_name, i))
        ax1.set_xlabel("turn-around time (us)")
        ax1.set_ylabel("the number of samples")
        ax1.set_xlim([0, max_value])
        ax1.hist(turnaround_times[i], bins=bins)

        plt.savefig("{}.part{}_histgram.pdf".format(session_name, i))

if __name__ == "__main__":
    # python elapsed_time_corr_dir.py dircetory_path
    parse_logs(sys.argv[1])
    visualize(session_name="behavior_velocity_planner")

