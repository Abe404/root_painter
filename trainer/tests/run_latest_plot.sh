#!/bin/bash

#python plot_metrics.py ./metrics f1 duration all_runs_f1_over_duration.png
#python plot_metrics.py ./metrics f1 epoch all_runs_f1_over_epoch.png



#python plot_metrics7.py ./metrics f1 duration adamw all_runs_f1_over_epoch.png

#python plot_metrics10.py ./metrics f1 duration adamw all_runs_f1_over_duration.png both
#python plot_metrics10.py ./metrics f1 duration adamw all_runs_f1_over_duration_adamw.png alternative
#python plot_metrics10.py ./metrics f1 duration adamw all_runs_f1_over_duration_sgd.png standard


python plot_metrics13.py ./metrics f1 duration baseline adamw both_runs_f1_over_duration.png both
python plot_metrics13.py ./metrics f1 duration baseline adamw baseline_runs_f1_over_duration.png baseline
python plot_metrics13.py ./metrics f1 duration baseline adamw adamw_runs_f1_over_duration.png adamw
xdg-open both_runs_f1_over_duration.png
