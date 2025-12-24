import numpy as np

tranch_dir = "datasets/ZINC12K.npy"
tranch_name = "ZINC12K"

tranch_dict = np.load(tranch_dir, allow_pickle=True).item()
tranch_dict_keys = list(tranch_dict.keys())

list_dict = {}
# Collect all numeric properties
for entry in tranch_dict[tranch_dict_keys[0]].keys():
    if type(tranch_dict[tranch_dict_keys[0]][entry]) == float or type(tranch_dict[tranch_dict_keys[0]][entry]) == int:
        list_dict[entry] = []

for smi in tranch_dict_keys:
    for prop in tranch_dict[smi].keys():
        if type(tranch_dict[smi][prop]) == float or type(tranch_dict[smi][prop]) == int:
            list_dict[prop].append(tranch_dict[smi][prop])

stats_dict = {}
print(f"Computing stats for properties: {list(list_dict.keys())}")

for entry in list_dict.keys():
    prop_list = list_dict[entry]
    # Use nanmean/nanstd to handle NaN values
    mean = np.nanmean(prop_list)
    std = np.nanstd(prop_list)
    
    stat = {}
    stat['mean'] = mean
    stat['std'] = std
    stats_dict[entry] = stat
    print(f"  {entry}: mean={mean:.4f}, std={std:.4f}")

output_path = f'datasets/{tranch_name}_stats.npy'
np.save(output_path, stats_dict)
print(f"\nSaved statistics to {output_path}")