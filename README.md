# Spatial Semantic Segmentation of Sound Scenes

This is a baseline implementation for the [DCASE2025 Challenge Task 4: Spatial Semantic Segmentation of Sound Scenes](https://dcase.community/challenge2025/task-spatial-semantic-segmentation-of-sound-scenes).

[DCASE2025 Challenge](https://dcase.community/challenge2025/index) provides an overview of the challenge tasks.


# Related Repositories
Part of `src/models/resunet` originates from  https://github.com/bytedance/uss/tree/master/uss/models \
Part of `src/models/m2dat` originates from  https://github.com/nttcslab/m2d \
Part of `src/modules/spatialscaper2` originates from  https://github.com/iranroman/SpatialScaper 


# Data and Environment Configuration
### Environment
Environment
```
git clone https://github.com/nttcslab/dcase2025_task4_baseline.git
cd dcase2025_task4_baseline

# Using conda
conda env create -f environment.yml
conda activate dcase2025t4

# Or using pip (python=3.11)
python -m venv dcase2025t4
source dcase2025t4/bin/activate
pip install -r requirements.txt
```
Install SpatialScaper
```
git clone https://github.com/iranroman/SpatialScaper.git
cd SpatialScaper
pip install -e .
```

SoX may be required for the above environment installation
```
sudo apt-get update && sudo apt-get install -y gcc g++ sox libsox-dev
```

### Checkpoints
M2D model checkpoint for the Audio Tagging model
```
cd dcase2025_task4_baseline
wget -P checkpoint https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip
unzip checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip -d checkpoint
```

### Data
DCASE2025Task4Dataset
```
# Download
# https://zenodo.org/10.5281/zenodo.15117227
UPDATING

cd dcase2025_task4_baseline
cp -r /path/to/DCASE2025Task4Dataset/* data/
```
Add data from other datasets
```
# Download Semantic Hearing's dataset
# https://github.com/vb000/SemanticHearing
wget -P data https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar

# Download EARS dataset using bash
# https://github.com/facebookresearch/ears_dataset
mkdir EARS
cd EARS
for X in $(seq -w 001 107); do
  curl -L https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip -o p${X}.zip
  unzip p${X}.zip
  rm p${X}.zip
done

# Add data
cd dcase2025_task4_baseline
bash add_data.sh --semhear_path /path/to/BinauralCuratedDataset --ears_path /path/to/EARS
```

### Verifying
```
cd dcase2025_task4_baseline
python verify.py --source_dir .
```

# Training
The system consists of two models, separation and audio tagging, which are trained separately.\
All the TensorBoard log and model checkpoint will be saved to `workspace`.
### Audio Tagging Model
The Audio Tagging model consists of a feature extractor backbone (M2D) and a head layer.\
We fine-tune the model in two steps:
```
# Train only the head
python -m src.train -c config/label/m2dat_head.yaml -w workspace/label

# Continue fine-tuning the last blocks of the M2D backbone, replace the BEST_EPOCH_NUMBER with the appropriate epoch number
python -m src.train -c config/label/m2dat_head_blks.yaml -w workspace/label -r workspace/label/m2dat_head/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
```

### Separation Model
There are two variants of the separation model, ResUNet and ResUNetK, which are trained using the following commands:
```
# ResUNet
python -m src.train -c config/separation/resunet.yaml -w workspace/separation

# ResUNetK
python -m src.train -c config/separation/resunetk.yaml -w workspace/separation
```


# Evaluating Baseline Checkpoints
There are three checkpoints for the two baseline systems, corresponding to the audio tagging model and two variants of the separation models described above.
These can be downloaded from the release [UPDATING] and placed in the `checkpoint` folder.

The two baseline systems can be evaluated on the `data/dev_set/test` data using:
```
# ResUNetK
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunetk.yaml
"""
CA-SDRi: 11.088
Label prediction accuracy: 59.80
"""

# ResUNet
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunet.yaml
"""
CA-SDRi: 11.032
Label prediction accuracy: 59.80
"""

# Evaluate and generate estimated waveforms
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunetk.yaml --generate_waveform
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunet.yaml --generate_waveform
```
To evaluate other model checkpoints, specify their paths under `tagger_ckpt` and `separator_ckpt` in the corresponding config files located in `src/evaluation/eval_configs`.
# Citation

If you use this system, please cite the following papers:

+ Binh Thien Nguyen, Masahiro Yasuda, Daiki Takeuchi, Daisuke Niizumi, Yasunori Ohishi, Noboru Harada, ”Baseline Systems and Evaluation Metrics for Spatial Semantic Segmentation of Sound Scenes,” in arXiv preprint arXiv 2503.22088, 2025, available at [URL](https://arxiv.org/abs/2503.22088).

+ Masahiro Yasuda, Binh Thien Nguyen, Noboru Harada, Romain Serizel, Mayank Mishra, Marc Delcroix, Shoko Araki, Daiki Takeuchi, Daisuke Niizumi, Yasunori Ohishi, Tomohiro Nakatani, Takao Kawamura, Nobutaka Ono, ”Description and discussion on DCASE 2025 challenge task 4: Spatial Semantic Segmentation of Sound Scenes,” in arXiv preprint arXiv:xxxx.xxxx, 2025, available at [URL]().
