# export LD_LIBRARY_PATH="/home/dhanajayb/anaconda3/envs/RELSO/lib::/usr/local/MATLAB/R2021a/bin/glnxa64/::$LD_LIBRARY_PATH"
# conda activate GRASSY

from GRASSY_utils import *

train_smi = []
with open("../datasets/FBAB.txt",'r') as f:
    for line in f:
	    train_smi.append(line.strip())

gen_smi = []
with open("ZINC_FBAB_gen.txt",'r') as f:
    for line in f:
	    gen_smi.append(line.strip())

print(fcd_score(train_smi, gen_smi))