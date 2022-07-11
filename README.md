#### GRASSY-Net

Hierarchial Generation of Molecules

##### Setup

`conda create --name DeepChem-torch`

`conda activate DeepChem-torch`

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

`pip install pandas`

`pip install 'deepchem[torch]'`

##### Training

`conda activate DeepChem-torch`

`cd hgraph2graph-master`

`python get_vocab.py --ncpu 32 < ../datasets/FBAB.txt > ZINC_FBAB_vocab.txt`

`python preprocess.py --train ../datasets/FBAB.txt --vocab ./ZINC_FBAB_vocab.txt --ncpu 32 --mode single`

`mkdir train_processed`

`mv tensor* train_processed/`

`mkdir ckpt/ZINC-FBAB-pretrained`

`python train_generator.py --train train_processed/ --vocab ./ZINC_FBAB_vocab.txt --save_dir ckpt/ZINC-FBAB-pretrained --save_iter 500 --seed 1337`

##### Molecule Generation

`python generate.py --vocab ./ZINC_FBAB_vocab.txt --model ./ckpt/ZINC-FBAB-pretrained/model.ckpt.1500 --nsample 100 > ZINC_FBAB_gen.txt`