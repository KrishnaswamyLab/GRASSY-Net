import numpy as np

tranch_dir = "processed/molecules.npy"
tranch_name = "molecules.npy"

tranch_dict = np.load(tranch_dir, allow_pickle=True).item()
tranch_dict_keys = list(tranch_dict.keys())

list_dict = {}
# import pdb; pdb.set_trace()
for entry in tranch_dict[tranch_dict_keys[0]].keys():
    if type(tranch_dict[tranch_dict_keys[0]][entry]) == float or type(tranch_dict[tranch_dict_keys[0]][entry]) == int:
        list_dict[entry] = []

for smi in tranch_dict_keys:
    for prop in tranch_dict[smi].keys():
        if type(tranch_dict[smi][prop]) == float or type(tranch_dict[smi][prop]) == int:
            list_dict[prop].append(tranch_dict[smi][prop])

stats_dict = {}
print(f"Prop list: {list_dict.keys()}")
for entry in list_dict.keys():
    prop_list = list_dict[entry]
    try:
        mean = np.mean(prop_list)
    except:
        print(f'Exception on property: {entry}')
        raise Exception
    std = np.std(prop_list)
    stat = {}
    stat['mean'] = mean
    stat['std'] = std
    stats_dict[entry] = stat

np.save(tranch_name + '_stats.npy', stats_dict)
