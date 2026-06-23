import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None


def _require_plotly():
    if px is None or go is None:
        raise ImportError("plotly is required for plotting helpers. Install it with `pip install plotly`.")

#Take assignmet and probabilities and return a dataframe with the probabilities for each group, as well as the final selected assignment's probabilities
def get_assignment_probs(assignment, ind_probs, groups=None):
	if groups is None:
		groups = assignment
	group_probs = {}
	group_errs = {}
	assignment_probs = {}
	for hid,intervention in assignment.items():
		reentry_prob = ind_probs.loc[hid,intervention]
		assignment_probs[hid] = reentry_prob
		group_probs.setdefault(groups[hid], []).append(reentry_prob)
	#sum each group's reentry probabilities
	for group, probs in group_probs.items():
		# print(group, probs)
		group_probs[group] = np.mean(probs)
		group_errs[group] = np.std(probs)
	return assignment_probs, group_probs, group_errs
	

def compare_assignments(assignments, names, ind_probs, groups=None, title='Probability of re-entry by group', plot = True):
	group_assignments = []
	group_errs_list = []
	assignment_probs_list = []
	totals = []
	for assignment, name in zip(assignments, names):
		assignment_probs, group_probs, group_errs = get_assignment_probs(assignment, ind_probs, groups=groups)
		total = np.mean(list(assignment_probs.values()))
		totals.append(total)
		group_assignments.append(group_probs)
		group_errs_list.append(group_errs)
		assignment_probs_list.append(assignment_probs)
	
	df_comp = pd.DataFrame(group_assignments).T
	df_comp.columns = names

	for assignment_probs, group_errs,name in zip(assignment_probs_list, group_errs_list, names):

		df_comp[f'{name} std'] = group_errs.values()
		totals.append(np.std(list(assignment_probs.values())))
	
	df_comp.loc['Total'] = totals

	#plot bar charts for comparison, with error bars"

	if plot:
		_require_plotly()
		fig = go.Figure()
		for name in names:
			fig.add_trace(go.Bar(x=df_comp.index, y=df_comp[name], name=name, error_y=dict(type='data', array=df_comp[f'{name} std'])))
		fig.update_layout(barmode='group', title=title)

		#label y axis as probability
		fig.update_yaxes(title_text='Probability of re-entry')
		fig.update_xaxes(title_text='Group')
		fig.show()
	
	return df_comp

def compare_groups_by_treatment(assignment_df):
	_require_plotly()
	#Get the fraction of households in each group that were assigned to each treatment
	treatments = assignment_df['Treatment'].unique()
	group_totals = assignment_df["Group"].value_counts().to_dict()
	groups = group_totals.keys()
	res_df = pd.DataFrame()
	for treatment in treatments:
		treatment_data = assignment_df[assignment_df['Treatment'] == treatment]
		group_counts = treatment_data["Group"].value_counts().to_dict()
		for group in group_counts.keys():
			group_counts[group] = group_counts[group]/group_totals[group]
		print(f'{treatment}: {group_counts}')
		res_df[treatment] = pd.Series(group_counts)
	
	#plot bars, stack by group
	fig = px.bar(res_df, barmode='stack')
	fig.show()

	return res_df

def get_groupwise_reentry_preds(df, groupname, probs):
	_require_plotly()
	#Get average probability of re-entry for each group
	average_probs = {}
	for group in df[groupname].unique():
		group_data = df[df[groupname] == group]
		group_probs = probs[probs['HouseholdID'].isin(group_data['HouseholdID'])].drop(columns=['HouseholdID', 'Outcome'])
		#take the mean across ES, SH, TH, Prev
		average_probs[group] = np.mean(list(group_probs.mean(axis=0).to_dict().values()))
	#plot
	fig = go.Figure()
	fig.add_trace(go.Bar(x=list(average_probs.keys()), y=list(average_probs.values())))
	fig.show()
	return average_probs

def gini(x, return_mean=False):
	#calculates the gini index of distribution x
	if not len(x):
		if return_mean:
			return 0, 0
		else:
			return 0
	x = np.array(x)
	diffsum = 0
	for i, xi in enumerate(x[:-1], 1):
		diffsum += np.sum(np.abs(xi - x[i:]))
	avg = np.mean(x)
	if avg == 0:
		return 0
	gini =  diffsum / (len(x)**2 * avg)
	if return_mean:
		return gini, avg
	else:
		return gini

def parse_dict(d):
	#Takes string representation of a dictionary and returns the dictionary
	#eg: "{1: 0.5, 2: 0.5}" -> {"1": 0.5, "2": 0.5}
	d = d[1:-1].split(',')
	d = [x.split(':') for x in d]
	d = {x[0].strip():float(x[1].strip()) for x in d}
	return d



class UtilityHandler:
	"""
	UtilityHandler is a base class for handling utility in the DECA framework.
	It provides methods to compute and manage utilities, including discounted utilities.
	Aggregation can be additive or averaged (parameterized by `aggregation_type`).
	Past discount and warm start parameters can be set to control the random initialization and discounting of utilities.
	"""
	def __init__(self, aggregation_type='additive', past_discount=0.995, warm_start=0):
		
		if aggregation_type not in ['additive', 'averaged']:
			raise ValueError(f"Invalid aggregation type: {aggregation_type}. Must be 'additive' or 'averaged'.")
		
		self.aggregation_type = aggregation_type
		self.past_discount = past_discount
		self.warm_start = warm_start

		self._time_step = None  # Used for averaged aggregation
		self._disc_time_step = None  # Used for averaged aggregation
	
	def get_system_utility(self, utils):
		"""
		Compute the system utility as the sum of utilities.
		"""
		if self.aggregation_type == 'additive':
			return np.sum(utils)
		elif self.aggregation_type == 'averaged':
			# multiply by time_step to get the true utilities, then sum
			return np.sum(utils) * self._time_step

	def reset_utils(self, n_agents):
		"""
		Initialize utilities in the environment.
		(True) Utilities are initialized to zero, and discounted utilities are set to zero as well.
		If warm_start is set, discounted utilities are initialized to a small random value within some distance of warm_start.
		"""		
		utils = np.zeros(n_agents, dtype=float)

		# Discounted utilities
		w = self.warm_start / 4
		disc_utils = np.array([
			self.warm_start + np.random.rand() * w - w/2
			for _ in range(n_agents)
		], dtype=float)
		
		# if averaged, pretend we've already seen one "fake" step, keep track of time_step and discounted_time_step
		if self.aggregation_type == "averaged":
			self._time_step = np.array([0. for _ in range(n_agents)])
			self._disc_time_step = np.array([0.1 for _ in range(n_agents)])

		return utils, disc_utils
	

	def _additive_update(self, utils, disc_utils, rewards):
		"""
		Additive update of utilities. Modifies the utils and disc_utils in place.
		"""
		assert len(utils) == len(rewards), "Utilities and rewards must have the same length."

		rewards = np.array(rewards, dtype=float)
		utils += rewards
		disc_utils *= self.past_discount
		disc_utils += rewards


	def _averaged_update(self, utils, disc_utils, rewards, counts=None, dummy_update=False):
		"""
		Update the utilities using time-based averaging.
		Counts is a dictionary of counts for each group, use this to increment the time_step, else increment by 1
		Needs to use the time_step attribute.
		IMPORTANT: The update should only be called when the time_step is incremented, otherwise the averaging will go out of sync
		TODO: Decide if the time_step should be incremented here or in the environment class. Doing it here for now.
		"""
		assert len(utils) == len(rewards), "Utilities and rewards must have the same length."
		if counts is None:
			new_time = self._time_step + 1
			new_disc_time = self._disc_time_step * self.past_discount + 1
		else:
			counts = np.array(counts, dtype=float)
			new_time = self._time_step + counts
			new_disc_time = self._disc_time_step * self.past_discount + counts
		rewards = np.array(rewards, dtype=float)

		# Handle 0 count indices
		skip_indices = new_time == 0
		new_time[skip_indices] = 1
		new_disc_time[skip_indices] += 1

		# Update utilities in place
		utils[:] = (utils * self._time_step + rewards) / new_time
		# Update discounted utilities in place
		disc_utils[:] = (disc_utils * self._disc_time_step + rewards) / new_disc_time

		# Handle 0 count indices, revert
		new_time[skip_indices] = 0
		new_disc_time[skip_indices] -=1

		# Update time steps
		if not dummy_update:
			self._time_step = new_time
			self._disc_time_step = new_disc_time


	def update_utilities(self, utils, disc_utils, rewards, counts=None, dummy_update=False):
		"""
		Update the utilities based on the true rewards.
		Ensure rewards do not include shaping rewards.
		"""

		if self.aggregation_type == 'additive':
			self._additive_update(utils, disc_utils, rewards)
		elif self.aggregation_type == 'averaged':
			self._averaged_update(utils, disc_utils, rewards, counts=counts, dummy_update=dummy_update)
		else:
			raise ValueError(f"Invalid aggregation type: {self.aggregation_type}. Must be 'additive' or 'averaged'.")
	
