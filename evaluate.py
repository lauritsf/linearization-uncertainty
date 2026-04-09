import logging

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata  # Import ContainerMetadata
from omegaconf.dictconfig import DictConfig  # Import DictConfig

from autograph.models.seq_models import SequenceModel

torch.serialization.add_safe_globals(
    [DictConfig, ContainerMetadata]
)  # Add DictConfig and ContainerMetadata to safe globals


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="test")
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Loading model from {cfg.model.pretrained_path}...")
    model = SequenceModel.load_from_checkpoint(cfg.model.pretrained_path, weights_only=False)
    model.update_cfg(cfg)

    datamodule = model._datamodule

    logger = [pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
