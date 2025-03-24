import sofa
from typing import Dict, List, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import importlib    

def initialize_config(module_cfg):
    module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else: return getattr(module, module_cfg["main"])()

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloader: dict,
        val_dataloader: dict = None,
    ):
        """Datamodule
    
        Args:
            train_dataloader (dict): {
                batch_size: int,
                num_workers: int,
                persistent_workers: bool
                dataset (dict): { # config for initialize dataset
                    module: path.to.module
                    main: MainFunction
                    args: {**kwargs}
                }
            }
            val_dataloader (dict, optional): similar to train
            }
        """
        super().__init__()
        self.train_config = train_dataloader
        self.val_config = val_dataloader

        self.train_dataset = initialize_config(self.train_config['dataset'])
        if val_dataloader is not None:            
            self.val_dataset = initialize_config(self.val_config['dataset'])
        else:
            self.val_dataset = None


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        pass
        
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_config['batch_size'],
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.train_config['num_workers'],
            pin_memory=True,
            persistent_workers=self.train_config['persistent_workers'],
            shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        if self.val_dataset is not None:
            val_dataloder = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.val_config['batch_size'],
                collate_fn=self.val_dataset.collate_fn,
                num_workers=self.val_config['num_workers'],
                pin_memory=True,
                persistent_workers=self.val_config['persistent_workers'],
                shuffle=False
            )
            return val_dataloder

    def test_dataloader(self):
        pass

    def teardown(self, stage=None):
        pass


if __name__ == '__main__':
    # cd ../..

    import os, sys; os.chdir('../..'); sys.path.append(os.getcwd())
    import yaml
    import importlib
    def initialize_config(module_cfg):
        module = importlib.import_module(module_cfg["module"])
        if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
        return getattr(module, module_cfg["main"])()

    config = {
        'module': 'src.datamodules.datamodule',
        'main': 'DataModule',
        'args': {
            'train_dataloader': {
                'batch_size': 4,
                'num_workers': 1,
                'persistent_workers': False,
                'dataset': {
                    "module": 'src.datamodules.dataset.semantic_hearing.curated_binaural_augrir_mono_wet',
                    'main': 'CuratedBinauralAugRIRDataset',
                    'args': {
                        "ref_channel": 0,
                        'fg_dir': '/home/nguyenbt/data/BinauralCuratedDataset/scaper_fmt/train',
                        'bg_dir': '/home/nguyenbt/data/BinauralCuratedDataset/TAU-acoustic-sounds/TAU-urban-acoustic-scenes-2019-development',
                        'bg_scaper_dir': '/home/nguyenbt/data/BinauralCuratedDataset/bg_scaper_fmt/train',
                        'jams_dir': '/home/nguyenbt/data/BinauralCuratedDataset/jams_hard/train',
                        'hrtf_dir': '/home/nguyenbt/data/BinauralCuratedDataset/hrtf',
                        'dset': 'train',
                        'sr': 44100,
                        'resample_rate': None,
                        'reverb': True
                    }
                }
            },
            'val_dataloader': {
                'batch_size': 4,
                'num_workers': 1,
                'persistent_workers': False,
                'dataset': {
                    "module": 'src.datamodules.dataset.semantic_hearing.curated_binaural_augrir_mono_wet',
                    'main': 'CuratedBinauralAugRIRDataset',
                    'args': {
                        "ref_channel": 0,
                        'fg_dir': '/home/nguyenbt/data/BinauralCuratedDataset/scaper_fmt/val',
                        'bg_dir': '/home/nguyenbt/data/BinauralCuratedDataset/TAU-acoustic-sounds/TAU-urban-acoustic-scenes-2019-development',
                        'bg_scaper_dir': '/home/nguyenbt/data/BinauralCuratedDataset/bg_scaper_fmt/val',
                        'jams_dir': '/home/nguyenbt/data/BinauralCuratedDataset/jams_hard/val',
                        'hrtf_dir': '/home/nguyenbt/data/BinauralCuratedDataset/hrtf',
                        'dset': 'val',
                        'sr': 44100,
                        'resample_rate': None,
                        'reverb': True
                    }
                }
            }
        }
    }
    datamodule = initialize_config(config)
