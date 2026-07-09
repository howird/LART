from typing import Any, Dict, Optional

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from lart.datamodules.components.hockey_action_dataset import HockeyPoseDataset


class HockeyDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig, train: bool = True):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val:
            if self.hparams.train:
                self.data_train = HockeyPoseDataset(self.hparams.cfg, split="train")
            self.data_val = HockeyPoseDataset(self.hparams.cfg, split="val")
        if not self.data_test:
            self.data_test = HockeyPoseDataset(self.hparams.cfg, split="test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.train_batch_size,
            num_workers=self.hparams.cfg.train_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.cfg.test_batch_size,
            num_workers=self.hparams.cfg.test_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.cfg.test_batch_size,
            num_workers=self.hparams.cfg.test_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass
