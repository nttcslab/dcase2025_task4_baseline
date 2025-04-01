# Related repositories
Part of `src/models/resunet` originates from  https://github.com/bytedance/uss/tree/master/uss/models \
Part of `src/models/m2dat` originates from  https://github.com/nttcslab/m2d \
Part of `src/modules/spatialscaper2` originates from  https://github.com/iranroman/SpatialScaper 


# Environment and data setting
### Environment
```
git clone https://github.com/nttcslab/dcase2025_task4_baseline.git
cd dcase2025_task4_baseline
conda env create -f environment.yml
conda activate dcase2025t4

git clone https://github.com/iranroman/SpatialScaper.git
cd SpatialScaper
pip install -e .
```

### Checkpoints
```
# M2D model checkpoint
cd dcase2025_task4_baseline
wget -P ckpt https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip
unzip ckpt/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip -d ckpt

# Baseline checkpoints
cd dcase2025_task4_baseline
wget -P ckpt UPDATING/m2dat.ckpt
wget -P ckpt UPDATING/resunet.ckpt
wget -P ckpt UPDATING/resunetk.ckpt
```

### Data
DCASE2025Task4Dataset
```
# Download
UPDATING

# Create a symlink to source directory
cd dcase2025_task4_baseline
ln -s $PATH_TO_DATA/DCASE2025Task4Dataset data
```
Add data from other dataset
```
# Download Semantic Hearing's data
# Download EARS

# ADD data
cd dcase2025_task4_baseline
bash add_data.sh --semhear_path /path/to/BinauralCuratedDataset --ears_path /path/to/EARS
```
### Verifying
```
cd dcase2025_task4_baseline
python verify.py --source_dir .
```

# Training
```
# Train separation model
python -m src.train -c config/separation/resunetk.yaml -w workspace/separation
python -m src.train -c config/separation/resunet.yaml -w workspace/separation

# Fine-tune label prediction model
python -m src.train -c config/label/m2dat_head.yaml -w workspace/label

# continue fine-tune the last blocks of the backbone, replace the BEST_EPOCH_NUMBER
python -m src.train -c config/label/m2dat_head_blks.yaml -w workspace/label -r workspace/label/m2dat_head/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
```

# Evaluating baseline checkpoint
```
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunetk.yaml
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunet.yaml

# Evaluate and generate estimated waveforms
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunetk.yaml --generate_waveform
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunet.yaml --generate_waveform
```


# References
UPDATING
