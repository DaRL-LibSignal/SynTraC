import numpy as np
import torch
import os
import json

def compare_accuracy(list1, list2):
	# Flatten both lists
	flat_list1 = [item for sublist in list1 for item in sublist]
	flat_list2 = [item for sublist in list2 for item in sublist]
	
	# Ensure both lists have the same number of elements
	if len(flat_list1) != len(flat_list2):
		return "Lists are of different sizes."
	
	# Count the number of matches
	matches = sum(1 for i, j in zip(flat_list1, flat_list2) if i == j)
	
	# Calculate accuracy
	accuracy = matches / len(flat_list1)
	return accuracy
		
class RL_Model:
	#Initialize the policy
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# policy = torch.jit.load("../policy_candidates/dt_try.pt").to(device)	

	@classmethod
	def get_model_output(cls, torch_input, policy):
		torch_input = (torch_input[None, :]).to(cls.device)
		action = policy(torch_input)
		# print(f"Action by policy {action}")
		return action
			
if __name__ == '__main__':
	case_1 = [ [1,0,0,0,0,0,0,0, 0,1,0,0],
			   [1,0,0,0,0,0,0,0, 0,0,1,0],
			   [1,0,0,0,0,0,0,0, 0,0,0,1],
			   [1,0,0,0,0,0,0,0, 1,0,0,0], 
			   [2,0,0,0,0,0,0,0, 0,1,0,0],
			   [2,0,0,0,0,0,0,0, 0,0,1,0],
			   [2,0,0,0,0,0,0,0, 0,0,0,1],
			   [2,0,0,0,0,0,0,0, 1,0,0,0],
			   [0,2,0,0,0,0,0,0, 0,1,0,0],
			   [0,2,0,0,0,0,0,0, 0,0,1,0],
			   [0,2,0,0,0,0,0,0, 0,0,0,1],
			   [0,2,0,0,0,0,0,0, 1,0,0,0],
			   [3,2,0,0,0,0,0,0, 0,1,0,0],
			   [3,2,0,0,0,0,0,0, 0,0,1,0],
			   [3,2,0,0,0,0,0,0, 0,0,0,1],
			   [3,2,0,0,0,0,0,0, 1,0,0,0],
			   [4,2,0,0,0,0,0,0, 1,0,0,0],
			   [3,2,0,0,0,0,0,0, 1,0,0,0],
			   [1,6,0,0,0,0,0,0, 1,0,0,0],
			   [2,2,0,0,0,0,0,0, 1,0,0,0],
			   [3,4,0,0,0,0,0,0, 1,0,0,0],
			   [1,5,0,0,0,0,0,0, 1,0,0,0],
			   [1,1,0,0,0,0,0,0, 1,0,0,0],
			   [9,9,0,0,0,0,0,0, 1,0,0,0]
			   ]
	case_2 =  [[0,0,1,0,0,0,0,0, 0,1,0,0],
			   [0,0,1,0,0,0,0,0, 0,0,1,0],
			   [0,0,1,0,0,0,0,0, 0,0,0,1],
			   [0,0,1,0,0,0,0,0, 1,0,0,0], 
			   [0,0,2,0,0,0,0,0, 0,1,0,0],
			   [0,0,2,0,0,0,0,0, 0,0,1,0],
			   [0,0,2,0,0,0,0,0, 0,0,0,1],
			   [0,0,2,0,0,0,0,0, 1,0,0,0],
			   [0,0,0,2,0,0,0,0, 0,1,0,0],
			   [0,0,0,2,0,0,0,0, 0,0,1,0],
			   [0,0,0,2,0,0,0,0, 0,0,0,1],
			   [0,0,0,2,0,0,0,0, 1,0,0,0],
			   [0,0,3,2,0,0,0,0, 0,1,0,0],
			   [0,0,3,2,0,0,0,0, 0,0,1,0],
			   [0,0,3,2,0,0,0,0, 0,0,0,1],
			   [0,0,3,2,0,0,0,0, 1,0,0,0],
			   [0,0,1,2,0,0,0,0, 0,1,0,0],
			   [0,0,1,5,0,0,0,0, 0,1,0,0],
			   [0,0,3,3,0,0,0,0, 0,1,0,0],
			   [0,0,0,7,0,0,0,0, 0,1,0,0],
			   [0,0,2,8,0,0,0,0, 0,1,0,0],
			   [0,0,2,2,0,0,0,0, 0,1,0,0],
			   [0,0,6,3,0,0,0,0, 0,1,0,0],
			   [0,0,4,4,0,0,0,0, 0,1,0,0]
			   ]
	case_3 =  [[0,0,0,0,1,0,0,0, 0,1,0,0],
			   [0,0,0,0,1,0,0,0, 0,0,1,0],
			   [0,0,0,0,1,0,0,0, 0,0,0,1],
			   [0,0,0,0,1,0,0,0, 1,0,0,0], 
			   [0,0,0,0,2,0,0,0, 0,1,0,0],
			   [0,0,0,0,2,0,0,0, 0,0,1,0],
			   [0,0,0,0,2,0,0,0, 0,0,0,1],
			   [0,0,0,0,2,0,0,0, 1,0,0,0],
			   [0,0,0,0,0,2,0,0, 0,1,0,0],
			   [0,0,0,0,0,2,0,0, 0,0,1,0],
			   [0,0,0,0,0,2,0,0, 0,0,0,1],
			   [0,0,0,0,0,2,0,0, 1,0,0,0],
			   [0,0,0,0,3,2,0,0, 0,1,0,0],
			   [0,0,0,0,3,2,0,0, 0,0,1,0],
			   [0,0,0,0,3,2,0,0, 0,0,0,1],
			   [0,0,0,0,3,2,0,0, 1,0,0,0],
			   [0,0,0,0,1,1,0,0, 0,0,1,0],
			   [0,0,0,0,1,3,0,0, 0,0,1,0],
			   [0,0,0,0,4,0,0,0, 0,0,1,0],
			   [0,0,0,0,3,5,0,0, 0,0,1,0],
			   [0,0,0,0,5,2,0,0, 0,0,1,0],
			   [0,0,0,0,1,2,0,0, 0,0,1,0],
			   [0,0,0,0,5,5,0,0, 0,0,1,0],
			   [0,0,0,0,7,4,0,0, 0,0,1,0]]
	case_4 =  [[0,0,0,0,0,0,1,0, 0,1,0,0],
			   [0,0,0,0,0,0,1,0, 0,0,1,0],
			   [0,0,0,0,0,0,1,0, 0,0,0,1],
			   [0,0,0,0,0,0,1,0, 1,0,0,0], 
			   [0,0,0,0,0,0,2,0, 0,1,0,0],
			   [0,0,0,0,0,0,2,0, 0,0,1,0],
			   [0,0,0,0,0,0,2,0, 0,0,0,1],
			   [0,0,0,0,0,0,2,0, 1,0,0,0],
			   [0,0,0,0,0,0,0,2, 0,1,0,0],
			   [0,0,0,0,0,0,0,2, 0,0,1,0],
			   [0,0,0,0,0,0,0,2, 0,0,0,1],
			   [0,0,0,0,0,0,0,2, 1,0,0,0],
			   [0,0,0,0,0,0,3,2, 0,1,0,0],
			   [0,0,0,0,0,0,3,2, 0,0,1,0],
			   [0,0,0,0,0,0,3,2, 0,0,0,1],
			   [0,0,0,0,0,0,3,2, 1,0,0,0],
			   [0,0,0,0,0,0,1,2, 0,0,0,1],
			   [0,0,0,0,0,0,2,3, 0,0,0,1],
			   [0,0,0,0,0,0,4,4, 0,0,0,1],
			   [0,0,0,0,0,0,5,2, 0,0,0,1],
			   [0,0,0,0,0,0,6,6, 0,0,0,1],
			   [0,0,0,0,0,0,3,7, 0,0,0,1],
			   [0,0,0,0,0,0,5,4, 0,0,0,1],
			   [0,0,0,0,0,0,7,6, 0,0,0,1]]
	
	# Specify the directory containing the .pt files
	folder_path = '/home/local/ASURITE/tchen169/Documents/Github-CV4TSC/CV4TSC/policy_candidates/'

	expected_output= [[1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0],
							  [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0],
							  [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0],
							  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0]]

	# Initialize a list to hold the loaded models
	loaded_models = {}
	# Iterate through each file in the folder
	for filename in os.listdir(folder_path):
		# Check if the file is a .pt file
		if filename.endswith('final.pt'):
			file_path = os.path.join(folder_path, filename)
			print("file path...................................", file_path)
			try:
				policy = torch.jit.load(file_path,  map_location='cuda')	
				# policy = torch.load(file_path,  map_location='cuda')	
			except Exception as E:
				policy = torch.load(file_path,  map_location='cuda')
			# Policy Loaded	

			action_list = []
			for i,case in enumerate([case_1, case_2, case_3, case_4]):
				output_action = []
				for j,input_vector in enumerate(case):
					##only for frap model
					# iv = input_vector[-4:]+input_vector[:-4]
					torch_input = torch.Tensor(input_vector).to(RL_Model.device)
					# print("torch input", torch_input)
					action = RL_Model.get_model_output(torch_input, policy)
					# print(f"Action by policy {action.item()}")
					output_action.append(action.item())
					print(f"{input_vector} R{action.item()} G{expected_output[i][j]}")
				action_list.append(output_action)


			'''EXPECTED OUTPUT = case 1[111011101110111000000000]
								case 2 [011101110111011100000000]
								case 3 [101110111011101100000000]
								case 4 [110111011101110100000000]'''
			# print(action_list)
			accuracy = compare_accuracy(action_list, expected_output)
			print("ACCURACY", accuracy)
			loaded_models[filename]=accuracy

	print(loaded_models)
	with open("accuracy.json", 'w') as json_file:
		json.dump(loaded_models, json_file, indent=4)


