import argparse
import logging
import yaml
import os

logger = logging.getLogger(__name__)


class ConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.config_parser = argparse.ArgumentParser(add_help=False)
        self.config_parser.add_argument(
            "-c",
            "--config",
            default=None,
            metavar="FILE",
            help="Where to load YAML configuration.",
        )
        self.option_names = []
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        """Same as :meth:`ArgumentParser.add_argument`."""
        arg = super().add_argument(*args, **kwargs)
        self.option_names.append(arg.dest)
        return arg

    def parse_args(self, args=None):
        """Same as :meth:`ArgumentParser.parse_args`."""
        res, remaining_argv = self.config_parser.parse_known_args(args)

        if res.config is not None:
            with open(res.config, "r") as f:
                config_vars = yaml.safe_load(f)
            for key in config_vars:
                assert (
                    key in self.option_names
                ), f"Unexpected configuration entry: {key}"
            self.set_defaults(**config_vars)

        return super().parse_args(remaining_argv)


def save_args(
    args: argparse.Namespace, filepath: str = "../config.yaml", rank: int = 0
) -> None:
    """If in master process, save ``args`` to a YAML file. Otherwise, do nothing.

    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        rank (int): Process rank in the distributed training. Defaults to 0.

    Example:
        >>> save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    """
    if isinstance(args, argparse.Namespace) and filepath.endswith(".yaml"):
        if rank != 0:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(args.__dict__, f)
        logger.info(f"Args is saved to {filepath}.")
        return None
    else:
        logger.warning(f"Args is not saved to {filepath}.")

        if not filepath.endswith(".yaml"):
            logger.warning(f"filepath {filepath} should end with .yaml.")
        if not isinstance(args, argparse.Namespace):
            logger.warning(f"args should be an argparse.Namespace but have {type(args)}.")
        return None
