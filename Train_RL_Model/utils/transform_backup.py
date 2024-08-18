import numpy as np
import pathlib
import json
import cv2
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import shutil
import math
import os


POS_MAP = {'A':0, 'B':1, 'C':2, 'D':3}
LIGHT_MAP = {'Green':1, 'Red':0, 'Yellow':0}

def one_hot(actions, max_idx):
	assert np.min(actions) != 0,'All red and yellow phase should not be included'
	actions_oh = np.zeros((actions.shape[0], max_idx), dtype=np.int8)
	actions_oh[np.arange(actions.shape[0]), actions-1] = 1
	return actions_oh

def process_meta_data(path):
	path_lib_data = pathlib.Path(path)
	path_lib_cur = pathlib.Path.cwd()
	path_lib_cur_metadata = pathlib.Path.joinpath(path_lib_cur, path_lib_data)
	path_lib_cur_data = path_lib_cur_metadata.parent
	processed_path = path.replace(path_lib_data.suffix, '_processed' + path_lib_data.suffix)
	if not path_lib_data.exists():
		raise IOError(f"No such json file under {path_lib_data}")
	else:
		if pathlib.Path(processed_path).exists():
			return processed_path
		with open(path_lib_data, 'r') as f:
			contents = json.load(f)
	processed = []
	for key, value in contents.items():
		for k in value.keys():
			if k == 'flag' or k == "Throughput":
				continue
			else:
				src = pathlib.Path(contents[key][k]['image'].split('cyclic_traffic_time_for_train/')[1])
				contents[key][k]['image'] = str(pathlib.Path.joinpath(path_lib_cur_data, src))
			# contents[key][k] = str(path_lib_cur_data)
	with open(processed_path, 'w') as pf:
		json.dump(contents, pf, indent=4)
	return processed_path

def render_box(d):
	thickness = 2
	name = r'{}'.format(d['image'])
	image = cv2.imread(name)
	for k in d.keys():
		if "Lane" in k:
			for v in d[k]['vehicles']:
				coordinates = [int(c) for c in v['coordinates']]
				image = cv2.rectangle(image, (coordinates[0],coordinates[1]),\
						   (coordinates[2], coordinates[3]), (0,255,0), thickness)
	return image

def read_image(d):
	name = r'{}'.format(d['image'])
	image = cv2.imread(name)
	return image

def cal_number(d):
	states = []
	rewards = []
	for k in d.keys():
		if "Lane" in k:
			# TODO: check order later
			states.append(d[k]['total_vehicles'])
			rewards.append(d[k]['stopped_vehicles'])
	return states, rewards

class Trajectory():
	def __init__(self, idx, src, dst, phases, acts, interval=10):
		self.interval = interval
		self.idx = idx
		self.src = src
		self.dst = dst
		self.phases = phases
		self.acts = acts
		self.processed = self._process()
		local_idx = []
		self.imgs = None
		self.cnts =  None
		self.light_phases = None
		self.actions =   None
		self.rewards =  None
		self.termials =  None
		self.timeouts = None
		for idx in range(len(self.processed)-1):
			# TODO: this is very heuristic, but should work fine in order to solve stochastic(at least to my current understanding) delay or extension of phase.
			divede_res = self._divide_helper(self.processed[idx], self.processed[idx+1])
			local_idx.extend(divede_res)
			# end point of last iteration is not considered
		local_idx.append(self.processed[idx+1])
		self.local_idx = [idx for idx in local_idx]
		self.global_idx = [self.src + idx for idx in local_idx]

	def _divide_helper(self, src, dst):
		# not include dst and self.interval should be multiple times of 10
		steps = round((dst - src - 1) / self.interval)
		_steps = [src]
		for i in range(steps-1):
			_steps.append(src + (i+1) * self.interval)
		return _steps
	
	def _process(self):
		# keep full action interval only
		slcs = []
		last = 0
		# filter out phase starting with 0
		for idx, i in enumerate(self.phases):
			if last == 0:
				if i != 0:
					last = i
					slcs.append(idx)
				continue
			else:
				if i == 0:
					start_point = idx
					slcs.append(idx)
					break
				else:
					continue
		last = 0
		for idx, phase in enumerate(self.phases[start_point:]):
			if phase == 0:
				if last != 0:
					slcs.append(idx+start_point)
			last = phase
		if last !=0:
			slcs.append(idx+start_point)
		# # TODO: rethink, action is taken at last obs then action changed into 0
		if slcs[0] == 0:
			slcs[1:] = [i-1 for i in slcs[1:]]
		else:
			slcs = [i-1 for i in slcs]
		return slcs

	def prepare_data(self, data_list, **kwargs):
		cnts = []
		imgs = []
		phases = []
		rewards = []
		actions = []
		pro_func = render_box if kwargs.get('detection') is True else read_image
		with tqdm(total=len(data_list)) as pbar:
			for i, pair in tqdm(enumerate(data_list[self.local_idx[0]:], start=self.local_idx[0])):
				# states is current states, phases is current phase (projected from no yellow and all red phases), action is next phase, rewards is averaged over all lanes and current time interval
				if i in self.local_idx:
					imageA = pro_func(pair[1]['A'])
					sA, rA = cal_number(pair[1]['A'])
					imageB = pro_func(pair[1]['B'])
					sB, rB = cal_number(pair[1]['B'])
					imageC = pro_func(pair[1]['C'])
					sC, rC = cal_number(pair[1]['C'])
					imageD = pro_func(pair[1]['D'])
					sD, rD = cal_number(pair[1]['D'])
					imgs.append(np.stack((imageA, imageB, imageC, imageD), axis=-1))
					phases.append(self.acts[i])
					cnts.append(np.concatenate((sA, sB, sC, sD), axis=0))
					rewards.append(0)
					try:
						
						rewards[-2] += (sum(rA+rB+rC+rD)/8)
						accumulation += 1
						actions.append(self.acts[i+1])
						rewards[-2] /= accumulation
						accumulation=0
					except IndexError:
						rewards.append(0)
						accumulation=0
						actions.append(self.acts[i+1])
				else:
					accumulation+=1
					_, rA = cal_number(pair[1]['A'])
					_, rB = cal_number(pair[1]['B'])
					_, rC = cal_number(pair[1]['C'])
					_, rD = cal_number(pair[1]['D'])
					# TODO: make it flexible
					rewards[-1] += (sum(rA+rB+rC+rD)/8)
			try:
				rewards[-1] /= accumulation
			except ZeroDivisionError:
				rewards.append(0)
		rewards = rewards[1:]
		timeouts = np.zeros(len(actions))
		timeouts[-1] = 1
		self.imgs = np.stack(imgs, axis=0).transpose(0,4,3,1,2)
		self.cnts = np.stack(cnts, axis=0)
		self.light_phases = np.array(phases)
		self.actions =  np.array(actions)
		self.rewards = np.array(rewards)
		self.terminals = np.zeros(len(actions))
		self.timeouts = timeouts

	def __str__(self):
		return f'From {self.src} to {self.dst}(Not included)'
	def __repr__(self):
		return f'From {self.src} to {self.dst}(Not included)'

def process_trajectories(data_list, detection):
	combination_list = []
	combination_set = set()
	timeouts_idx = []
	idx_list = []
	for k, c in data_list:
		# lights = [((idx, light['traffic_light']) for idx, light in c.items())]
		lights = 0
		if c['flag'] == 1:
			timeouts_idx.append(int(k))
		for idx, l in list(c.items())[2:]:
			lights += (LIGHT_MAP[l['traffic_light']] << POS_MAP[idx])
			# [((idx, light['traffic_light']) for idx, light in c.values())]
			# lights.sort(key=lambda x: x[0])
			# combination_list.append([l[1] for l in lights])
		idx_list.append(k)
		combination_list.append(lights)
		combination_set.add(lights)
	timeouts_idx.append(len(data_list))
	sorted_cmbntns = sorted(list(combination_set))
	phases = [sorted_cmbntns.index(a) for a in combination_list]
	acts = phase2act(phases)
	traj_list = []
	# for i in range(len(phases)):
	# 	print(phases[i], acts[i])
	for i in range(len(timeouts_idx[:-1])):
		cur_traj = Trajectory(i, timeouts_idx[i], timeouts_idx[i+1], phases[timeouts_idx[i]: timeouts_idx[i+1]], acts[timeouts_idx[i]: timeouts_idx[i+1]])
		traj_list.append(cur_traj) #data_list[timeouts_idx[i]: timeouts_idx[i+1]+1]
		cur_traj.prepare_data(data_list[cur_traj.src: cur_traj.dst+1], detection=detection)
	return traj_list

def process_datapoints(data_list, detection):
	pro_func = render_box if detection else read_image
	imgs = []
	cnts = []
	with tqdm(total=len(data_list)) as pbar:
		for i, pair in tqdm(enumerate(data_list)):
			# states is current states, phases is current phase (projected from no yellow and all red phases), action is next phase, rewards is averaged over all lanes and current time interval
			imageA = pro_func(pair[1]['A'])
			sA, _ = cal_number(pair[1]['A'])
			imageB = pro_func(pair[1]['B'])
			sB, _ = cal_number(pair[1]['B'])
			imageC = pro_func(pair[1]['C'])
			sC, _ = cal_number(pair[1]['C'])
			imageD = pro_func(pair[1]['D'])
			sD, _ = cal_number(pair[1]['D'])
			imgs.append(np.stack((imageA, imageB, imageC, imageD), axis=-1))
			cnts.append(np.concatenate((sA, sB, sC, sD), axis=0))
			pbar.update(1)
	imgs = np.array(imgs)
	cnts = np.array(cnts)
	return imgs, cnts

def phase2act(phases):
	acts = []
	start = 0
	for i in phases[::-1]:
		if i == 0:
			acts.append(start)
		else:
			start = i
			acts.append(start)
	acts.reverse()
	return acts

def process_raw_data(data_path, detection=True):
	# for safety, rearrange this into
	final_dir = f'./Dataset/detection_{detection}'
	if os.path.exists(final_dir):
		print('Finished data loading')
		return 0
	with open(data_path, 'r') as f:
		data_dict = json.load(f)
	if Path('./Dataset/tmp').exists():
		shutil.rmtree('./Dataset/tmp')
	Path.mkdir(Path('./Dataset/tmp'))
	data_list = [(key, value) for key, value in data_dict.items()]
	data_list.sort(key=lambda x:float(x[0]))
	# TODO: add more flexibilitys
	traj_list = process_trajectories(data_list, detection)
	for idx, traj in enumerate(traj_list):
		np.savez(f'./Dataset/tmp/process_traj_{idx}.npz', imgs=traj.imgs, cnts=traj.cnts, phases=traj.light_phases, rewards=traj.rewards, actions=traj.actions, timeouts=traj.timeouts, terminals=traj.terminals)
	shutil.copytree('./Dataset/tmp', final_dir)
	shutil.rmtree('./Dataset/tmp')
	print('Finished data generation')
	return 1

def process_actions(contents):
	combination_list = []
	combination_set = set()
	timeouts_idx = []
	for k, c in contents:
		# lights = [((idx, light['traffic_light']) for idx, light in c.items())]
		lights = 0	
		if c['flag'] == 1:
			timeouts_idx.append(int(k))
		for idx, l in list(c.items())[2:]:
			lights += (LIGHT_MAP[l['traffic_light']] << POS_MAP[idx])
			# [((idx, light['traffic_light']) for idx, light in c.values())]
			# lights.sort(key=lambda x: x[0])
			# combination_list.append([l[1] for l in lights])
		combination_list.append(lights)
		combination_set.add(lights)
	sorted_cmbntns = sorted(list(combination_set))
	actions = [sorted_cmbntns.index(a) for a in combination_list]
	return actions, sorted_cmbntns

def down_sampling2change_phase(states, rewards, actions, terminals, timeouts):
	change_idx = []
	action_space = np.max(actions)
	for idx in range(1, len(actions)):
		# First element is not 0
		if actions[idx] == 0 and actions[idx-1] != 0:
			change_idx.append(idx - 1)
	states_ds = states[change_idx, :]
	rewards_ds = accumulate_rewards(rewards, change_idx)
	actions_ds = actions[change_idx]
	# actions_ds = one_hot(actions_ds, action_space)
	return states_ds, rewards_ds, actions_ds

# def down_sampling1act_taken(row_data, action_interval=10):
# 	# return index only

def accumulate_rewards(rewards, idxes):
	idxes = deepcopy(idxes)
	idxes.insert(0, 0)
	agg_rewards = []
	for i in range(1, len(idxes)):
		agg_rewards.append(np.sum(rewards[idxes[i-1]:idxes[i]]))
	return np.array(agg_rewards)

if __name__ == "__main__":
	processed = process_meta_data('./Dataset/Intersection_camera.json')
	process_code = process_raw_data(processed, detection=True)
	print('Trajectories generated')
