#!/bin/bash
# Independent cross-checks of the V2 results against raw Deribit prints (no LEAN involved).
# Low CPU priority so the workstation stays usable. Outputs go to results/analysis/.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
P="nice -n 19 /home/czf/miniconda3/bin/python"
mkdir -p $W/results/analysis
cd $W/analysis
$P indep_weekly.py   > $W/results/analysis/indep_weekly.txt 2>&1      # writes results/indep_weekly.pkl
$P reconcile.py      > $W/results/analysis/reconcile.txt 2>&1         # needs indep_weekly.pkl
$P lean_vs_prints.py > $W/results/analysis/lean_vs_prints.txt 2>&1
$P vrp_horizons.py   > $W/results/analysis/vrp_horizons.txt 2>&1
$P hedged_costs.py   > $W/results/analysis/hedged_costs.txt 2>&1
echo ANALYSIS_DONE >> $W/results/analysis/hedged_costs.txt
