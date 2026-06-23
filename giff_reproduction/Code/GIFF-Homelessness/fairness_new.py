from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Tuple
from utils import get_assignment_probs, gini, UtilityHandler

##### SI-X methods #####
def fair_score2(groups, groups_assigned_probs, beta):
    # Get average probs for each group
    group_probs = {}
    group_counts = {}
    for group in groups:
        group_probs[group] = np.mean(groups_assigned_probs[groups_assigned_probs['Group'] == group]['Probs'])
        group_counts[group] = len(groups_assigned_probs[groups_assigned_probs['Group'] == group]['Probs'])
    # Get the average of the group probs, ignore nans
    # prevent mean of empty slice error
    # Get the average of the group probs, ignore nans
    vals = np.array(list(group_probs.values()), dtype=float)

    if vals.size == 0 or np.all(np.isnan(vals)):
        avg_prob = np.nan   # or 0.0, depending on what makes sense for you
    else:
        avg_prob = np.nanmean(vals)

    # avg_prob = np.nanmean(list(group_probs.values())) 
    if np.isnan(avg_prob):
        avg_prob = 0
    #Set nan=avg
    for group in groups:
        if np.isnan(group_probs[group]):
            group_probs[group] = avg_prob +0.01 #If a  group hasnt been seen, assume it is slightly worse than average. Benefit of doubt
    # print(avg_prob, group_probs)
    # Get score for each group
    group_scores = {}
    for group in groups:
        group_score = beta*(avg_prob - group_probs[group])
        group_scores[group] = (group_score, group_probs[group], group_counts[group]) #Score, average group prob, group count
    #Remove nans
    for group in groups:
        if np.isnan(group_scores[group][0]):
            group_scores[group] = (0, group_probs[group], group_counts[group])
    return group_scores

def update_intervention_probs2(row, all_treatments=None):
    #Using the new formulation: group score is multiplied by the change this assignment would cause in the group's average probability
    if all_treatments is None:
        print('Error: all_treatments must be specified')
        exit()
    
    interventions = deepcopy(all_treatments)
    group_score, group_prob, group_count = row['Groupscore']
    probs = []
    for intervention in interventions:
        probs.append(row[intervention])
    probs = np.array(probs)

    #Group scores have the form (score, avg_prob, count)
    #update probs, scale by group score
    for i, intervention in enumerate(interventions):
        # Calculate change in group average by intervention
        delta_g = (group_prob*group_count + probs[i])/(group_count+1) - group_prob #Faces scaling issues, as with time, counts become larger, and the change becomes smaller
        delta_g = probs[i] - group_prob #Unscaled. Just measure the distance from the group average
        row[intervention] = probs[i] - group_score*delta_g
    return row


# -----------------------------
# Utilities
# -----------------------------

def _proportional_counts_for_window(
    counts_running: Dict[str, int], needed: int
) -> Dict[str, int]:
    """Scale the remaining counts proportionally so they sum to `needed`.

    If rounding leaves us short, we top-up greedily from treatments that still have
    remaining capacity, to ensure the total matches `needed`.
    """
    total_left = sum(counts_running.values())
    if total_left <= 0:
        return {k: 0 for k in counts_running}

    # Initial proportional allocation
    out = {
        k: int(np.floor(needed * (v / total_left)))
        for k, v in counts_running.items()
    }
    # Top-up to match `needed`
    deficit = needed - sum(out.values())
    if deficit > 0:
        # Greedily allocate where capacity exists
        pool = sorted(counts_running.items(), key=lambda kv: -kv[1])
        i = 0
        while deficit > 0 and any(v > 0 for _, v in pool):
            k, v = pool[i % len(pool)]
            if counts_running[k] - out[k] > 0:
                out[k] += 1
                deficit -= 1
            i += 1
    return out


def _finalize_counts_after_window(
    counts_running: Dict[str, int],
    counts_current: Dict[str, int]
) -> None:
    for k in counts_current:
        counts_running[k] = max(0, counts_running[k] - counts_current[k])


##### GIFF methods #####
def giff_qf_row(
    row: pd.Series,
    gidx: int,
    zs_pre,
    zg_count,
    overall_group_count,
    fairness_function,
    wg=None,
) -> pd.Series:
    """
    Compute a GIFF fairness row for one household:
      - Keep 'pre' state fixed (zs_pre, disc_pre).
      - For each treatment column in `row`, simulate a local update:
            z_g <- GIFF_update(z_g; add reward = row[col], counts(g)=1)
        and compute ΔF = F_post - F_pre.
      - Return a Series aligned to `row.index`.

    Args:
        row:          pd.Series with Q_u values for this household (only treatment columns).
        gidx:         int group index for this household.
        zs_pre:       current payoff vector (any mutable type `util_handler` accepts).
        disc_pre:     current discounted/aux state for `util_handler`.
        util_handler: your UtilityHandler instance.
        fairness_function: callable like np.var or your F.

    Returns:
        pd.Series with ΔF per treatment (same index as `row`).
    """
    # If household has no valid group, return zeros
    if gidx is None:
        return pd.Series(0.0, index=row.index, dtype=float)

    z = np.asarray(zs_pre, dtype=float)
    # NOTE: This assumes zi=0 means no data for that group
    # if everything is 0, set everything to average of the current row rewards
    if np.all(z == 0):
        z = np.mean(row.values.astype(float))
        z = np.full_like(zs_pre, z) # Shape as z_pre
    
    # print("Z: ", z)
    # if the entry for this group is 0, set it to the average of Z + 0.01
    if z[gidx] == 0:
        z[gidx] = np.nanmean(z) + 0.01 # benefit of doubt
    # set all zeros to the average of Z
    z[z == 0] = np.nanmean(z)
    
    F_pre = fairness_function(z)
    z_g  = float(z[gidx])
    N_g  = float(zg_count)
    r    = row.values.astype(float)             # shape (T,)
    z_g_new = (N_g * z_g + r) / (N_g + 1.0)     # shape (T,)
    out = np.empty_like(r)
    z_buf = z.copy()
    for k, zgn in enumerate(z_g_new):
        z_buf[gidx] = zgn
        out[k] = float(fairness_function(z_buf)) - F_pre

    
    dF = out
    if wg is not None:
        scale = (N_g + 1)/(N_g + max(1, wg))
        dF = dF * scale

    return pd.Series(dF, index=row.index, dtype=float)


# -----------------------------
# Cost builders (per time-window)
# -----------------------------

def build_costs_SI(
    *,
    tw_probs: pd.DataFrame,
    group_info: pd.Series,  # index: HouseholdID, values: Group
    groups: Iterable[str],
    assigned_groups_so_far: pd.DataFrame,  # columns: HouseholdID, Group, Probs
    beta: float,
    all_treatments: List[str],
    maximize: bool,
) -> pd.DataFrame:
    """Build per-household cost matrix for SI.

    SI uses `fair_score2` to derive a scalar group score, then applies
    `update_intervention_probs2` row-wise to adjust Q_u before assignment.
    """
    # Compute group scores from the assignment history so far
    group_scores = fair_score2(list(groups), assigned_groups_so_far, beta)

    # Annotate and transform
    tw_mod = tw_probs.copy()
    tw_mod['Group'] = tw_mod.index.map(group_info)
    tw_mod['Groupscore'] = tw_mod['Group'].map(group_scores)

    tw_mod = tw_mod.apply(
        update_intervention_probs2,
        axis=1,
        args=(all_treatments,),
    )
    cost = tw_mod[all_treatments]
    if maximize:
        cost = 1.0 - cost
    return cost


def build_costs_GIFF(
    *,
    tw_probs: pd.DataFrame,
    group_info: pd.Series,  # index: HouseholdID -> Group
    group_to_idx: Dict[str, int],
    zs: np.ndarray,                 # running per-group averages (shape [G])
    ng_counts: np.ndarray,          # running per-group counts (shape [G])
    overall_group_counts: Dict[str, int],
    beta: float,
    all_treatments: List[str],
    fairness_function: Callable[[np.ndarray], float],
    maximize: bool,
) -> pd.DataFrame:
    """Build per-household cost matrix for GIFF.

    We keep the current state fixed (zs, ng_counts), and for each candidate
    action compute ΔF via `giff_qf_row`. Final per-household cost is
    (1-β) * Q_u  ±  β * Q_f (sign depends on maximize).
    """
    # Base utility matrix (Q_u): costs to minimize (e.g., re-entry probs)
    Q_u = tw_probs[all_treatments].copy()
    if maximize:
        Q_u = 1.0 - Q_u

    Q_f = Q_u.copy()

    # Window group counts for scaling (wg)
    tw_groups = tw_probs.index.to_series().map(group_info)
    tw_group_counts = tw_groups.value_counts().to_dict()

    # Precompute ΔF per household. Modify in place
    # Use *current* zs/Ng for all households in this window
    z_pre = np.asarray(zs, dtype=float)
    for h in Q_u.index:
        g = group_info.get(h, None)
        gidx = group_to_idx.get(g, None)
        if gidx is None:
            Q_f.loc[h] = 0.0
            continue
        Q_f.loc[h] = giff_qf_row(
            row=Q_u.loc[h],
            gidx=gidx,
            zs_pre=z_pre,
            zg_count=float(ng_counts[gidx]),
            overall_group_count=overall_group_counts[g],
            fairness_function=fairness_function,
            wg=float(tw_group_counts.get(g, 0)),
        )

    Q_f = Q_f*10000
    # Combine (sign depends on maximize)
    if maximize:
        cost = (1.0 - beta) * Q_u + beta * Q_f
    else:
        cost = (1.0 - beta) * Q_u - beta * Q_f
    return cost


# -----------------------------
# Unified temporal loop
# -----------------------------

def temporal_assignment_unified(
    *,
    df: pd.DataFrame,
    probs: pd.DataFrame,
    assignment_method,
    groupname: str,
    beta: float,
    all_treatments: List[str],
    method: str = 'SI',                # 'SI' | 'GIFF'
    constrained: bool = False,
    maximize: bool = False,
    fairness_function: Callable[[np.ndarray], float] | None = None,
    **solver_kwargs,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Single pass over time windows that delegates to either SI or GIFF.

    Returns
    -------
    overall_assignment : Dict[HouseholdID, Treatment]
    zs                 : np.ndarray of per-group running averages (final)
    ng_counts          : np.ndarray of per-group counts (final)
    """
    assert method in {"SI", "GIFF"}
    if method == "GIFF":
        if fairness_function is None:
            raise ValueError("fairness_function must be provided for GIFF")

    # Setup
    group_info = df.set_index('HouseholdID')[[groupname]].copy().rename(columns={groupname: 'Group'})
    group_info_series = group_info['Group'].fillna('Unknown')   
    groups = group_info_series.unique().tolist()             
    group_to_idx = {g: i for i, g in enumerate(groups)}

    # Global treatment capacities
    counts_total = probs.Original.value_counts().to_dict()
    counts_running = counts_total.copy()

    # Per-group running stats (zs = running average of chosen Q_u; ng_counts = counts)
    G = len(groups)
    zs = np.zeros(G, dtype=float)
    ng_counts = np.zeros(G, dtype=float)

    overall_assignment: Dict = {}
    overall_group_counts = group_info_series.value_counts().to_dict()

    # For SI history-based scoring
    assigned_groups = pd.DataFrame(columns=['HouseholdID', 'Group', 'Probs'])

    # Iterate time windows in order
    num_windows = len(df['TimeWindow'].unique())
    for tw in range(num_windows):
        # Households in this TW
        tw_data = df[df['TimeWindow'] == tw].copy()
        if tw_data.empty:
            continue

        # Probabilities for this TW
        tw_probs = probs[probs['HouseholdID'].isin(tw_data['HouseholdID'])].copy()
        tw_probs = tw_probs.set_index('HouseholdID')
        tw_probs = tw_probs[all_treatments].copy().fillna(0.0)

        # Decide available counts for this TW
        if constrained:
            needed = len(tw_probs)
            counts_current = _proportional_counts_for_window(counts_running, needed)
            # Ensure feasibility
            if sum(counts_current.values()) != needed:
                raise RuntimeError("Internal count allocation error: mismatch with needed assignments")
        else:
            counts_current = counts_running  # allocate against the global remainder directly

        # Build costs matrix per method
        if method == 'SI':
            cost_mat = build_costs_SI(
                tw_probs=tw_probs,
                group_info=group_info_series,
                groups=groups,
                assigned_groups_so_far=assigned_groups,
                beta=beta,
                all_treatments=all_treatments,
                maximize=maximize,
            )
        elif method == 'GIFF':  # GIFF
            cost_mat = build_costs_GIFF(
                tw_probs=tw_probs,
                group_info=group_info_series,
                group_to_idx=group_to_idx,
                zs=zs,
                ng_counts=ng_counts,
                overall_group_counts=overall_group_counts,
                beta=beta,
                all_treatments=all_treatments,
                fairness_function=fairness_function,
                maximize=maximize,
            )

        # Solve assignments for this TW
        tw_assignment, updated_counts = assignment_method(
            cost_mat, counts_current, maximize=maximize, **solver_kwargs
        )
        overall_assignment.update(tw_assignment)

        # If constrained, decrement counts_running by counts_current used.
        # If unconstrained, the solver already decremented its passed-in dict
        # (by contract), so we refresh from the returned value.
        if constrained:
            _finalize_counts_after_window(counts_running, counts_current)
        else:
            counts_running = updated_counts

        # Commit *running* group stats with ORIGINAL Q_u (pre-maximize transform)
        # tw_probs is already indexed by HouseholdID; use it for scalar lookups
        for h, t in tw_assignment.items():
            g = group_info_series.get(h, None)   # Series.get uses index (HouseholdID)
            if g is None:
                continue
            gidx = group_to_idx[g]

            # Scalar cell (no FutureWarning)
            q_u = tw_probs.loc[h, t]

            # incremental average
            n = ng_counts[gidx]
            zs[gidx] = (n * zs[gidx] + q_u) / (n + 1.0)
            ng_counts[gidx] = n + 1.0

        # Update SI history table using ORIGINAL probs for the chosen intervention
        tw_groups = tw_data[['HouseholdID']].copy()
        tw_groups['Group'] = tw_groups['HouseholdID'].map(group_info_series)       # Series mapper
        tw_groups['Intervention'] = tw_groups['HouseholdID'].map(tw_assignment)

        # Scalar per-row lookup from tw_probs (already indexed by HouseholdID)
        tw_groups['Probs'] = tw_groups.apply(
            lambda r: tw_probs.loc[r['HouseholdID'], r['Intervention']],
            axis=1,
        )
        
        if assigned_groups.empty:
            assigned_groups = tw_groups
        else:
            assigned_groups = pd.concat(
                [assigned_groups, tw_groups[['HouseholdID', 'Group', 'Probs']]], ignore_index=True
            )

    return overall_assignment, zs, ng_counts


# -----------------------------
# Simple sweep wrapper
# -----------------------------

def temporal_beta_sweep_unified(
    df: pd.DataFrame,
    probs: pd.DataFrame,
    groupname: str,
    betas: Iterable[float],
    solver,
    all_treatments: List[str],
    *,
    method: str = 'SI',
    constrained: bool = False,
    maximize: bool = False,
    fairness_function: Callable[[np.ndarray], float] | None = None,
    verbose: bool = True,
    **solver_kwargs,
) -> pd.DataFrame:
    """Run a β-sweep using the unified temporal loop.

    Returns a results DataFrame with columns: [Beta, Gini, Total, Group_probs]
    """
    ind_probs = probs.set_index('HouseholdID').copy()
    ind_probs = ind_probs.drop(columns=['Original'], errors='ignore')

    group_dict = df.set_index('HouseholdID')[groupname].to_dict()

    results_df = pd.DataFrame(columns=['Beta', 'Gini', 'Total', 'Group_probs'])

    for beta in betas:
        assignment, zs, ng = temporal_assignment_unified(
            df=df,
            probs=probs,
            assignment_method=solver,
            groupname=groupname,
            beta=beta,
            all_treatments=all_treatments,
            method=method,
            constrained=constrained,
            maximize=maximize,
            fairness_function=fairness_function,
            **solver_kwargs,
        )

        assignment_probs, group_probs, _ = get_assignment_probs(assignment, ind_probs, groups=group_dict)
        total = float(np.mean(list(assignment_probs.values()))) if assignment_probs else np.nan
        beta_gini = float(gini(list(group_probs.values()))) if group_probs else np.nan

        print(f"Beta: {beta}, Gini: {beta_gini}, Total: {total}")

        row = pd.DataFrame([
            {'Beta': beta, 'Gini': beta_gini, 'Total': total, 'Group_probs': group_probs}
        ])
        if not results_df.empty:
            results_df = pd.concat([results_df, row], ignore_index=True)
        else:
            results_df = row

    return results_df
