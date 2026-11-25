"""Reads --exp from the command line and sets it before config.py is imported.

Import this module FIRST in a script:   import exp_arg  # noqa
"""
import argparse
import os

_p = argparse.ArgumentParser()
_p.add_argument("--exp", default="scene", choices=["scene", "motion", "mixed25"],
                help="which experiment's classes to use (default: scene)")
os.environ["AR_EXP"] = _p.parse_args().exp
