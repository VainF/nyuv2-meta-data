wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip

python extract_nyuv2.py --mat nyu_depth_v2_labeled.mat --normal_zip nyuv2_surfacenormal_metadata.zip  --data_root NYUv2 --save_colored