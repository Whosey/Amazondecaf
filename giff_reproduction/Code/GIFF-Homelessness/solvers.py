from copy import deepcopy

try:
    from gurobipy import Model, GRB, quicksum
except ImportError:
    Model = None
    GRB = None
    quicksum = None

#Get ILP assignment
#Use gurobi to solve the ILP
def get_ILP_assignment(ind_probs, counts, all_treatments=None, maximize=False):
    if Model is None:
        return get_ILP_assignment_ORTools(
            ind_probs,
            counts,
            all_treatments=all_treatments,
            maximize=maximize,
        )

    if all_treatments is None:
        all_treatments = list(ind_probs.columns)
        if 'Outcome' in all_treatments:
            all_treatments.remove('Outcome')

    counts = counts.copy()
    #Create model
    m = Model("ILP")
    treatments = deepcopy(all_treatments)
    #Create variables
    # x = m.addVars(ind_probs.index, ind_probs.columns, vtype=GRB.BINARY, name="x")
    x = m.addVars(ind_probs.index, treatments, vtype=GRB.BINARY, name="x")

    #Create objective
    if maximize:
        m.setObjective(quicksum(x[(hid,intervention)]*ind_probs.loc[hid, intervention] for hid in ind_probs.index for intervention in treatments), GRB.MAXIMIZE)
    else:
        m.setObjective(quicksum(x[(hid,intervention)]*ind_probs.loc[hid, intervention] for hid in ind_probs.index for intervention in treatments), GRB.MINIMIZE)

    #Create constraints
    #Each household must be assigned to exactly one intervention
    m.addConstrs((quicksum(x[(hid,intervention)] for intervention in treatments) == 1 for hid in ind_probs.index), "c1")
    #Each intervention must have at most the number of slots available
    m.addConstrs((quicksum(x[(hid,intervention)] for hid in ind_probs.index) <= counts[intervention] for intervention in treatments), "c2")

    #Optimize model. Suppress output
    m.setParam('OutputFlag', False)
    m.optimize()

    # print('Obj: %g' % m.objVal)

    #Get assignment
    assignment = {}
    for hid in ind_probs.index:
        for intervention in treatments:
            if x[(hid,intervention)].x == 1:
                assignment[hid] = intervention
                break
    #update counts
    for intervention in treatments:
        counts[intervention] -= sum(x[(hid,intervention)].x for hid in ind_probs.index)
    return assignment, counts

from ortools.sat.python import cp_model

def get_ILP_assignment_ORTools(ind_probs, counts, all_treatments=None, maximize=False):
    # Select treatments
    if all_treatments is None:
        all_treatments = list(ind_probs.columns)
        if 'Outcome' in all_treatments:
            all_treatments.remove('Outcome')

    counts = counts.copy()
    treatments = deepcopy(all_treatments)
    hids = list(ind_probs.index)

    # CP-SAT model
    model = cp_model.CpModel()

    # Decision vars: x[(hid, tr)] in {0,1}
    x = {(hid, tr): model.NewBoolVar(f"x_{hid}_{tr}") for hid in hids for tr in treatments}

    # Each household assigned to exactly one treatment
    for hid in hids:
        model.Add(sum(x[(hid, tr)] for tr in treatments) == 1)

    # Each treatment has at most the available slots
    for tr in treatments:
        cap = int(counts[tr])
        model.Add(sum(x[(hid, tr)] for hid in hids) <= cap)

    # CP-SAT expects integer coefficients, so scale floating point costs.
    scale = 1_000_000
    objective = sum(
        int(round(float(ind_probs.loc[hid, tr]) * scale)) * x[(hid, tr)]
        for hid in hids
        for tr in treatments
    )
    if maximize:
        model.Maximize(objective)
    else:
        model.Minimize(objective)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools CP-SAT failed with status {status}")

    # Extract assignment
    assignment = {}
    for hid in hids:
        chosen = None
        for tr in treatments:
            if solver.Value(x[(hid, tr)]) == 1:
                chosen = tr
                break
        if chosen is None:  # fallback (shouldn't happen under ==1)
            chosen = max(treatments, key=lambda t: solver.Value(x[(hid, t)]))
        assignment[hid] = chosen

    # Update counts
    for tr in treatments:
        used = sum(1 for hid in hids if assignment[hid] == tr)
        counts[tr] -= used

    return assignment, counts


#For each time window, solve the ILP problem
#Get the assignment for each household
def solve_temporal_assignment(assignment_method, probs, df_tw, num_windows):
    """
    df_tw: dataframe with 'TimeWindow' column that indicates the time window for each household
    """
    counts_total = probs.Original.value_counts().to_dict()
    counts_running = counts_total.copy()
    overall_assignment = {}
    for tw in range(num_windows):
        #Get the data for that time window
        tw_data = df_tw[df_tw['TimeWindow'] == tw]
        #Get the probabilities for that time window
        tw_probs = probs[probs['HouseholdID'].isin(tw_data['HouseholdID'])]
        #Index the probabilities by household ID
        tw_probs.set_index('HouseholdID', inplace=True)
        #Get the assignment for that time window
        tw_assignment, counts_running = assignment_method(tw_probs, counts_running)
        overall_assignment.update(tw_assignment)

    return overall_assignment
