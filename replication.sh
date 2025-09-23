#!/bin/bash

set -e

#
# Preliminaries
#

# Install dependencies
pip install -r requirements.txt

# Build Cython module
python setup_cython.py build_ext --inplace
python -c "import model_cython; print('Cython ready')"

#
# Table 1: Sparsity Analysis
#
# Table 1: benchmark_results/tab_sparsity.tex

python run_benchmarks.py --parallel --sparsity

#
# Figures 2 and 3: Value Function Algorithms
#
# Figure 2, Panel (a): vf_algorithms.pdf
# Figure 2, Panel (b): vf_algorithms_zoom.pdf
# Figure 3: vf_rate_analysis.pdf

python vf_algorithms.py

#
# Tables 2 and 3, Figures 4 and 5: Monte Carlo results
#
# Table 2: mc_results/tab_mc_7x5_1000.tex
# Table 3: mc_results/tab_mc_7x5_4000.tex
# Figure 4: mc_results/boxplot-7x5-1000.pdf
# Figure 5: mc_results/computational-analysis-7x5-1000.pdf

python run_mc.py --n_players 7 --n_demand 5 --n_mc 100 --n_obs 1000,4000 --model model --output_dir mc_results

#
# Table 4: Matrix Exponential Comparison
#
# Table 4: benchmark_results/tab_matrix_expm.tex

python run_benchmarks.py --parallel --expm

#
# Multiple Equilibria Check
#

python check_mc_equilibria.py
