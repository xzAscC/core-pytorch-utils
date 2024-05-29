from .loss_buffer import HistoryBuffer
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from .lr_scheduler import LRWarmupScheduler
from .logger import setup_logger
from typing import List, Dict, Optional, Tuple
import time
import os
import weakref
import numpy as np
from .hooks.hookbase import HookBase
from .hooks.lr_update_hook import LRUpdateHook
from .hooks.checkpoint_hook import CheckpointHook
from .hooks.logger_hook import LoggerHook
from .misc import set_random_seed, collect_env, symlink
from torch.cuda.amp import GradScaler, autocast

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


class Trainer:
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
        self.train_by_epoch = max_epochs > 0
        self.model = model
        epoch_len = len(data_loader) if self.train_by_epoch else None
        self.optimizer = optimizer
        self.lr_scheduler = LRWarmupScheduler(
            lr_scheduler,
            by_epoch,
            epoch_len,
            warmup_t,
            warmup_by_epoch,
            warmup_mode,
            warmup_init_lr,
            warmup_factor,
        )
        self.data_loader = data_loader
        self.unpack_batch_dict = unpack_batch_dict
        self.work_dir = work_dir
        self.metric_storage = MetricStorage()

        if self.train_by_epoch:
            self.epoch_len = len(data_loader)
            self.max_epochs = max_epochs
            self.max_iters = max_epochs * epoch_len
        else:
            self.max_iters = max_iters

        self.cur_iter = 0
        self.start_iter = 0

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._max_num_checkpoints = max_num_checkpoints
        self._checkpoint_period = checkpoint_period
        self._log_period = log_period
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp

        self._default_setup()

    @property
    def inner_iter(self) -> int:
        """The iteration within the epoch, ranged in [0, epoch_len - 1]."""
        assert (
            self.train_by_epoch
        ), "inner_iter is only available when training by epoch."
        return self.cur_iter % self.epoch_len

    @property
    def cur_epoch(self) -> int:
        """The current epoch, ranged in [0, max_epochs - 1]."""
        assert (
            self.train_by_epoch
        ), "cur_epoch is only available when training by epoch."
        return self.cur_iter // self.epoch_len

    @property
    def ckpt_dir(self) -> str:
        """The directory to save checkpoints. Overwrite this method to change the path."""
        return os.path.join(self.work_dir, "checkpoints")

    @property
    def tb_log_dir(self) -> str:
        """The directory to save tensorboard files. Overwrite this method to change the path."""
        return os.path.join(self.work_dir, "tb_logs")

    @property
    def hook_info(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ + f" (priority {h.priority})" for h in self._hooks]

    def log(self, *args, **kwargs) -> None:
        """Update the metrics stored in :obj:`self.trainer.metric_storage`."""
        self.metric_storage.update(*args, **kwargs)

    def _default_setup(self) -> None:
        setup_logger("template", output_dir=self.work_dir, rank=0)

        logger.info("Environment info:\n" + collect_env())

        default_hooks = [LRUpdateHook()]
        default_hooks.extend(
            [
                CheckpointHook(self._checkpoint_period, self._max_num_checkpoints),
                LoggerHook(self._log_period, tb_log_dir=self.tb_log_dir),
            ]
        )
        self.register_hooks(default_hooks)
        logger.info(f"Registered default hooks: {self.hook_info}")

        self._grad_scaler = GradScaler(enabled=self._enable_amp)
        if self._enable_amp:
            logger.info("Automatic Mixed Precision (AMP) training is on.")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        split_line = "-" * 50
        logger.info(
            f"\n{split_line}\n"
            f"Work directory: {self.work_dir}\n"
            f"Checkpoint directory: {self.ckpt_dir}\n"
            f"Tensorboard directory: {self.tb_log_dir}\n"
            f"{split_line}"
        )

    def register_hooks(self, hooks: List[HookBase]) -> None:
        """Register hooks to the trainer.

        Args:
            hooks (list[HookBase]): List of hooks to be registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: HookBase) -> None:
        """Register a hook to the trainer.

        For hooks with the same priority, they are executed in the order they are registered.

        Args:
            hook (HookBase): The hook to be registered.
        """
        assert isinstance(hook, HookBase)
        assert hook.priority >= 1 and hook.priority <= 10
        # To avoid circular reference, hooks and trainer cannot own each other. This normally
        # does not matter, but will cause memory leak if the involved objects contain __del__.
        # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        hook.trainer = weakref.proxy(self)
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if hook.priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _log_iter_metrics(
        self, loss_dict: Dict[str, torch.Tensor], data_time: float, iter_time: float
    ) -> None:
        """
        Args:
            loss_dict (dict): Dict of scalar losses.
            data_time (float): Time taken by the dataloader iteration.
            iter_time (float): Time taken by one complete iteration.
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict.update(data_time=data_time, iter_time=iter_time)
        # gather metrics among all workers for logging
        all_metrics_dict = [metrics_dict]

        self.log(self.cur_iter, lr=self.lr, smooth=False)

        # data_time among workers can have high variance. The actual latency
        # caused by data_time is the maximum among workers.
        data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
        self.log(self.cur_iter, data_time=data_time)

        # same as data_time
        iter_time = np.max([x.pop("iter_time") for x in all_metrics_dict])
        self.log(self.cur_iter, iter_time=iter_time)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict])
            for k in all_metrics_dict[0].keys()
        }
        losses_reduced = sum(metrics_dict.values())
        if not np.isfinite(losses_reduced):
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration={self.cur_iter}! "
                f"loss_dict={metrics_dict}."
            )

        self.log(self.cur_iter, total_loss=losses_reduced)
        if len(metrics_dict) > 1:
            self.log(self.cur_iter, **metrics_dict)

    def train_one_iter(self) -> None:
        """Train one iteration.

        Subclass :class:`cpu.trainer.Trainer` and implement your own :meth:`train_one_iter`
        to do something fancier.
        """
        iter_start_time = time.perf_counter()

        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            batch = next(self._data_iter)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        # If self._enable_amp=False, autocast and GradScaler’s calls become no-ops.
        # This allows switching between default precision and mixed precision
        # without if-else statements.
        with autocast(enabled=self._enable_amp):
            if self.unpack_batch_dict:
                loss_dict = self.model(**batch)
            else:
                loss_dict = self.model(batch)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        ##########################
        # 3. Calculate gradients #
        ##########################
        self.optimizer.zero_grad()
        self._grad_scaler.scale(losses).backward()
        if self._clip_grad_norm > 0:
            self._grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

        ##############################
        # 4. Update model parameters #
        ##############################
        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()

        self._log_iter_metrics(
            loss_dict, data_time, time.perf_counter() - iter_start_time
        )

    def train(
        self, resume_from_checkpoint: Optional[str] = None, auto_resume: bool = True
    ) -> None:
        """Start training.

        If ``resume_from_checkpoint`` is specified, resume from the given checkpoint.
        Otherwise, auto resume from the latest checkpoint.

        Args:
            resume_from_checkpoint (str): Path to the checkpoint. Defaults to None.
            auto_resume (bool): Defaults to True.
        """
        if resume_from_checkpoint is not None:
            self.load_checkpoint(path=resume_from_checkpoint)
        else:
            self.load_checkpoint(auto_resume=auto_resume)

        logger.info(f"Start training from iteration {self.start_iter}")

        self._call_hooks("before_train")
        for self.cur_iter in range(self.start_iter, self.max_iters):
            if self.train_by_epoch and self.cur_iter % self.epoch_len == 0:
                self._call_hooks("before_epoch")
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
            if self.train_by_epoch and (self.cur_iter + 1) % self.epoch_len == 0:
                self._call_hooks("after_epoch")
        self._call_hooks("after_train")

    def save_checkpoint(self, file_name: str) -> None:
        """Save training state: ``epoch``, ``num_gpus``, ``model``, ``optimizer``, ``lr_scheduler``,
        ``metric_storage``, ``hooks`` (optional), ``grad_scaler`` (optional).

        Args:
            filename (str): The checkpoint will be saved as ``ckpt_dir/filename``.
        """
        data = {
            "num_gpus": 1,
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage,
        }
        data.update(
            dict(epoch=self.cur_epoch)
            if self.train_by_epoch
            else dict(iter=self.cur_iter)
        )
        hook_states = {
            h.class_name: h.state_dict() for h in self._hooks if h.checkpointable
        }
        if hook_states:
            data["hooks"] = hook_states
        if self._enable_amp:
            data["grad_scaler"] = self._grad_scaler.state_dict()

        file_path = os.path.join(self.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag the latest checkpoint
        dst_file = os.path.join(self.ckpt_dir, "latest.pth")
        symlink(file_name, dst_file)

    def load_checkpoint(self, path: Optional[str] = None, auto_resume: bool = False):
        """Load the given checkpoint or resume from the latest checkpoint.

        Args:
            path (str): Path to the checkpoint to load.
            auto_resume (bool): If True, automatically resume from the latest checkpoint.
        """
        if path is None and auto_resume:
            latest_ckpt = os.path.join(self.ckpt_dir, "latest.pth")
            if not os.path.exists(latest_ckpt):
                logger.warning(
                    "You specify auto_resume=True, but we fail to find "
                    f"{latest_ckpt} to auto resume from."
                )
            else:
                logger.info(f"Found {latest_ckpt} to auto resume from.")
                path = latest_ckpt
        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")
        else:
            logger.info("Skip loading checkpoint.")
            return

        # check if the number of GPUs is consistent with the checkpoint
        num_gpus = 0
        ckpt_num_gpus = checkpoint["num_gpus"]
        assert num_gpus == ckpt_num_gpus, (
            f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
            f"but currently only have {num_gpus} GPUs."
        )

        # 1. load epoch / iteration
        if self.train_by_epoch:
            start_epoch = checkpoint["epoch"] + 1
            self.start_iter = start_epoch * self.epoch_len
        else:
            self.start_iter = checkpoint["iter"] + 1

        # 2. load model
        incompatible = self.model_or_module.load_state_dict(
            checkpoint["model"], strict=False
        )
        if incompatible.missing_keys:
            logger.warning(
                "Encounter missing keys when loading model weights:\n"
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "Encounter unexpected keys when loading model weights:\n"
                f"{incompatible.unexpected_keys}"
            )

        # 3. load metric_storage
        self.metric_storage = checkpoint["metric_storage"]

        # 4. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 5. load lr_scheduler
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 6. load grad scaler
        consistent_amp = not (self._enable_amp ^ ("grad_scaler" in checkpoint))
        assert (
            consistent_amp
        ), "Found inconsistent AMP training setting when loading checkpoint."
        if self._enable_amp:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        # 7. load hooks
        hook_states = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(
                f"Encounter missing keys when loading hook state dict:\n{missing_keys}"
            )
        if unexpected_keys:
            logger.warning(
                f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}"
            )

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break

    @property
    def model_or_module(self) -> nn.Module:
        """The model not wrapped by :class:`DistributedDataParallel`."""
        return self.model


class MetricStorage(dict):
    """The class stores the values of multiple metrics (some of them may be noisy, e.g., loss,
    batch time) in training process, and provides access to the smoothed values for better logging.

    The class is designed for automatic tensorboard logging. User should specify the ``smooth``
    when calling :meth:`update`, so that we can determine which metrics should be
    smoothed when performing tensorboard logging.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss will be smoothed, but lr will not
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}
        >>> # like dict, can be indexed by string
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: Dict[str, HistoryBuffer] = self
        self._smooth: Dict[str, bool] = {}
        self._latest_iter: Dict[str, int] = {}

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (int): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (bool): If True, return the smoothed values of these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same ``smooth`` in different calls to :meth:`update`.
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
                self._history[key] = HistoryBuffer(window_size=self._window_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key]
                self._latest_iter[key] = iter
            else:
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """Return the smoothed values or the latest values of multiple metrics.
        The specific behavior depends on the ``smooth`` when updating metrics.

        Returns:
            dict[str -> (int, float)]:
                Mapping from metric name to its (the latest iteration, the avg / the latest value)
                pair.
        """
        return {
            key: (self._latest_iter[key], his_buf.avg if self._smooth[key] else his_buf.latest)
            for key, his_buf in self._history.items()
        }
