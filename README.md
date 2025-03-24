### References
UPDATING

### Environments
```
conda create -n evn_name python=3.11
conda activate env_name
pip install -r requirements.txt

git clone https://github.com/iranroman/SpatialScaper.git
cd SpatialScaper
pip install -e .
```

Download M2D-AT checkpoint
```
cd dcase2025_task4_baseline
wget -P ckpt https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip
unzip ckpt/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip -d ckpt
```

### Data
UPDATING




### Training
```
# Train separation model
python -m src.train -c config/separation/resunetk.yaml -w workspace/separation

# Fine-tune label prediction model
python -m src.train -c config/label/m2dat_head.yaml -w workspace/label

# continue fine-tune the last two blocks of the backbone, replace the BEST_EPOCH_NUMBER
python -m src.train -c config/label/m2dat_head_2blks.yaml -w workspace/label -r workspace/label/m2dat_head/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
```
