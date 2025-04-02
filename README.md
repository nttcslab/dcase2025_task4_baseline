# Spatial Semantic Segmentation of Sound Scenes

This is a baseline implementation for the [DCASE2025 Challenge Task 4: Spatial Semantic Segmentation of Sound Scenes](https://dcase.community/challenge2025/task-spatial-semantic-segmentation-of-sound-scenes).

[DCASE2025 Challenge](https://dcase.community/challenge2025/index) provides an overview of the challenge tasks.

## Description
### Systems
The system consists of two models, audio tagging (AT) and source separation (SS), which are trained separately.
The AT model consists of a pre-trained feature extractor backbone (M2D) and a head layer.
For SS, we provide two variants: ResUNet and ResUNetK.

### Data folder structure
The data consists of two parts: the Development dataset and the Evaluation dataset.
The Evaluation dataset will be released at a later stage.
The structure of the Development dataset is as follows:
```
data
`-- dev_set
    |-- config
    |   |-- EARS_config.json
    |   `-- FSD50K_config.json
    |-- metadata
    |   |-- valid
    |   `-- valid.json
    |-- noise
    |   |-- train
    |   `-- valid
    |-- room_ir
    |   |-- train
    |   `-- valid
    |-- sound_event
    |   |-- train
    |   `-- valid
    `-- test
        |-- oracle_target
        `-- soundscape
```
The `config`, `metadata`, `noise`, `room_ir`, and `sound_event` folders are used for generating the training data, including the train and validation splits.\
The `test` folder contains the test data for evaluating the model checkpoints, including the pre-mixed soundscapes in `soundscape` and the oracle target sources in `oracle_target`.


### Related Repositories
Part of `src/models/resunet` originates from  https://github.com/bytedance/uss/tree/master/uss/models \
Part of `src/models/m2dat` originates from  https://github.com/nttcslab/m2d \
Part of `src/modules/spatialscaper2` originates from  https://github.com/iranroman/SpatialScaper 


## Data Preparation and Environment Configuration
### Setting
Clone repository
```
git clone https://github.com/nttcslab/dcase2025_task4_baseline.git
cd dcase2025_task4_baseline
```
Install environment
```
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

### Data Preparation
The Development dataset can be donwloaded and placed into `data` folder as
```
# Download and unzip all files from https://zenodo.org/10.5281/zenodo.15117227
zip -s 0 DCASE2025Task4Dataset.zip --out unsplit.zip
unzip unsplit.zip

# Place the dev_set in dcase2025_task4_bas/data folder
ln -s "$(pwd)/final_0402_1/DCASE2025Task4Dataset/dev_set" /path/to/dcase2025_task4_bas
eline/data
```
In addition to the recorded data, sound events are also added from other dataset as
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

Verifying data folder structure
```
cd dcase2025_task4_baseline
python verify.py --source_dir .
```

## Training

All the TensorBoard log and model checkpoint will be saved to `workspace`.
### Audio Tagging Model
Before training, checkpoint of the M2D model should be downloaded as
```
cd dcase2025_task4_baseline
wget -P checkpoint https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip
unzip checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip -d checkpoint
```

AT model is fine-tuned in two steps:
```
# Train only the head
python -m src.train -c config/label/m2dat_head.yaml -w workspace/label

# Continue fine-tuning the last blocks of the M2D backbone, replace the BEST_EPOCH_NUMBER with the appropriate epoch number
python -m src.train -c config/label/m2dat_head_blks.yaml -w workspace/label -r workspace/label/m2dat_head/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
```

### Separation Model
Two variants of the separation model, ResUNet and ResUNetK, are trained using:
```
# ResUNet
python -m src.train -c config/separation/resunet.yaml -w workspace/separation

# ResUNetK
python -m src.train -c config/separation/resunetk.yaml -w workspace/separation
```


## Evaluating Baseline Checkpoints
There are three checkpoints for the two baseline systems, corresponding to the ATg model and two variants of the SS models described above.
These can be downloaded from the release [UPDATING] and placed in the checkpoint folder.

Class-aware Signal-to-Distortion Ratio (CA-SDRi) and label prediction accuracy can be calculated on the data/dev_set/test data using the baseline checkpoints as
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
