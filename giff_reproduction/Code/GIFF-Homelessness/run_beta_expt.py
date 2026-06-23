import numpy as np
import pandas as pd
from pathlib import Path
from solvers import get_ILP_assignment
from fairness_new import temporal_beta_sweep_unified
from utils import gini

#LOAD DATA
use_PSH = False
if use_PSH:
    #Using Nguyen data (with PSH)
    probs = pd.read_csv('Data/dat_all_preds_bayestree_foundkids_plusPSH.csv', index_col=0)
    #Rename "Real" column to "Original", 'Reentered' to 'Outcome'
    probs.rename(columns={'Real':'Original', 'Reentered':'Outcome'}, inplace=True)
    #Real to original mappping
    intervention_maps = {1: 'ES', 2: 'PSH', 3: 'TH', 4: 'RRH', 5: 'PREV'}
    #Replace the numbers with the actual treatment names
    probs.replace({'Original': intervention_maps}, inplace=True)
    all_treatments = ['ES', 'PSH', 'TH', 'RRH', 'PREV']

else:
    probs = pd.read_csv('Data/BARTprobs.csv', index_col=0)
    intervention_maps  = {0: 'ES', 1: 'TH', 2: 'RRH', 3:'Prev'}
    all_treatments = ['ES', 'TH', 'RRH', 'Prev']
household_data = pd.read_csv('Data/homelessData.csv', index_col=0)

HID = probs.HouseholdID
HIDN = household_data.HouseholdIdentificationNumber #HIDN is the common column key between the two dataframes
#Rename the HouseholdIdentificationNumber column household data to HouseholdID
household_data.rename(columns={'HouseholdIdentificationNumber':'HouseholdID'}, inplace=True)

#join the two dataframes by HouseholdID
df = pd.merge(probs, household_data, on='HouseholdID', how='inner')

#use household ID as index in probs
ind_probs = probs.set_index('HouseholdID', inplace=False)
ind_probs.drop('Original', axis=1, inplace=True)
#find probability of household 200001080 being assigned to intervention ES
# x = ind_probs.loc[200001080, 'ES']

#get counts of each interventaion available overall
counts = df.Original.value_counts().to_dict()
original_assignment = df[['Original','HouseholdID']]
original_assignment = original_assignment.set_index('HouseholdID').to_dict()['Original']

#Divide data into time steps based on entry time
df['EntryDate'] = pd.to_datetime(df['EntryDate'])
df['ExitDate'] = pd.to_datetime(df['ExitDate'])

#Assume a 30-day window
day_window_size = 30
start_date = df['EntryDate'].min()
end_date = df['EntryDate'].max()
num_days = (end_date - start_date).days
num_windows = int(num_days/day_window_size)

#divide data into time windows
df['TimeWindow'] = pd.cut(df['EntryDate'], bins=num_windows, labels=range(num_windows))
df['TimeWindow'].value_counts()
df.sort_values(by=['TimeWindow','EntryDate'], inplace=True)

sel_cols = []
lens = []
for col in df.columns:
    uniques = df[col].unique()
    if len(uniques) < 20:
        uniques = df[col].unique()
        final_uniques = []
        #Remove all unique values which have less than 10 rows with that value
        for unique in uniques:
            if df[df[col] == unique].shape[0] < 50:
                continue
            else:
                final_uniques.append(unique)
        if len(final_uniques) > 1:
            sel_cols.append(col)
            lens.append(len(final_uniques))

#plot histogram of lens
print(len(lens))
# fig = px.histogram(x=lens, nbins=20)
# fig.show()
if 'Original' in sel_cols:
    sel_cols.remove('Original')
if 'Outcome' in sel_cols:
    sel_cols.remove('Outcome')

print(sel_cols)

import argparse

parser = argparse.ArgumentParser(description="Run beta experiment with configurable arguments.")
parser.add_argument('--method', type=str, default='GIFF', choices=['SI', 'GIFF'], help='Assignment method to use.')
parser.add_argument(
    '--groups',
    nargs='*',
    default=None,
    help='Optional group columns to run. Example: --groups Children Ethnicity',
)

args = parser.parse_args()
if args.groups:
    missing = sorted(set(args.groups) - set(sel_cols))
    if missing:
        raise ValueError(f"Unknown group columns: {missing}")
    sel_cols = args.groups
    print(f"Running selected groups: {sel_cols}")

method = args.method
constrained = True
if method == 'SI':
    betas = [0, 10,25,50,75,100,250,500,750,1000,2500,5000,7500,10000]
if method == 'GIFF':
    betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9995, 0.9999, 1.0]

solver = get_ILP_assignment
results = pd.DataFrame()
for i,groupname in enumerate(sel_cols):
    print(i,groupname)
    res = temporal_beta_sweep_unified(
        df, probs, groupname, betas, solver, all_treatments, 
        constrained=constrained,
        method=method,
        fairness_function=lambda x: -gini(x),
        maximize=False,
        )
    
    res['Group'] = groupname
    res['Time Window Size'] = day_window_size
    results = pd.concat([results, res])


# savename
save_name = ""
mod = ""
if method == 'SI':
    save_name = "SI-X"
if method == 'GIFF':
    save_name = "GIFF"
dirname = "Experiments/"
if constrained:
    dirname = dirname + "Constrained/"
else:
    dirname = dirname + "All_Interventions/"
Path(dirname).mkdir(parents=True, exist_ok=True)
save_name = dirname + save_name + mod
results.to_csv(save_name + ".csv")

