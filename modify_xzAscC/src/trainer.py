import logging
from typing import List

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """An epoch-base trainer. It assumes that every step, we:
    1. Load a batch from the data_loader.
    2. Compute the loss with the batch.
    3. Compute the gradients with the above loss.
    4. Update the model with the optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        data_loader: DataLoader,
        unpack_batch_dict: bool = False,
        max_epochs: int = 0,
        max_iters: int = 0,
        work_dir: str = "work_dir",
        max_num_checkpoints: int = None,
        checkpoint_period: int = 1,
        log_period: int = 50,
        clip_grad_norm: float = 0.0,
        enable_amp: bool = False,
        by_epoch: bool = True,
        warmup_t: int = 0,
        warmup_by_epoch: bool = False,
        warmup_mode: str = "fix",
        warmup_init_lr: float = 0.0,
        warmup_factor: float = 0.0,
    ):
        model.train()
        assert (max_epochs > 0) ^ (
            max_iters > 0
        ), "Please specify either max_epochs or max_iters."
        self.train_by_epoch = max_epochs > 0

        self.model = model
        self.optimizer = optimizer

        self.data_loader = data_loader
        self.unpack_batch_dict = unpack_batch_dict
        self.work_dir = work_dir
        self.metric_storage = MetricStorage()

        if self.train_by_epoch:
            self.epoch_len = len(data_loader)
            self.max_epochs = max_epochs
            self.max_iters = self.max_epochs * self.epoch_len
        else:
            self.max_iters = max_iters
            self.epoch_len = None

        self.lr_scheduler = LRWarmupScheduler(
            lr_scheduler,
            by_epoch,
            self.epoch_len,
            warmup_t,
            warmup_by_epoch,
            warmup_mode,
            warmup_init_lr,
            warmup_factor,
        )

        self.cur_iter = 0  # [0, max_iters - 1]
        self.start_iter = 0  # [0, max_iters - 1]

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._max_num_checkpoints = max_num_checkpoints
        self._checkpoint_period = checkpoint_period
        self._log_period = log_period
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp

        self._default_setup()
