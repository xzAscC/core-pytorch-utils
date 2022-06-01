import os
import argparse
import yaml
from copy import deepcopy
from argparse import Namespace
from typing import Optional, List


class ConfigArgumentParser(argparse.ArgumentParser):
    """Argument parser that supports loading a YAML configuration file.

    A small issue: config file values are processed using :meth:`ArgumentParser.set_defaults`
    which means ``required`` and ``choices`` are not handled as expected. For example, if you
    specify a required value in a config file, you still have to specify it again on the
    command line.

    If this issue matters, the `ConfigArgParse <http://pypi.python.org/pypi/ConfigArgParse>`_
    library can be used as a substitute.
    """

    def __init__(self, *args, **kwargs):
        self.config_parser = argparse.ArgumentParser(add_help=False)
        self.config_parser.add_argument(
            "-c", "--config", default=None, metavar="FILE",
            help="where to load YAML configuration"
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
                assert key in self.option_names, f"Unexpected configuration entry: {key}"
            self.set_defaults(**config_vars)

        return super().parse_args(remaining_argv)


def save_args(
    args: Namespace,
    filepath: str,
    excluded_fields: Optional[List[str]] = None,
    rank: int = 0
) -> None:
    """If in master process, save ``args`` to a YAML file. Otherwise, do nothing.

    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        excluded_fields (list[str]): Names of the fields that are not saved.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert isinstance(args, Namespace)
    assert filepath.endswith(".yaml")
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    save_dict = deepcopy(args.__dict__)
    for field in excluded_fields or []:
        save_dict.pop(field)
    with open(filepath, "w") as f:
        yaml.dump(save_dict, f)
    print(f"[cpu.config_parser] Args is saved to {filepath}.")
