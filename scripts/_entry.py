"""Small script adapters for the package CLI (use the installed package)."""
import argparse

from crosssense.cli import main


def run(command: str) -> int:
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('--config', default='configs/default.yaml')
    args, remainder = cli.parse_known_args()
    return main(['--config', args.config, command, *remainder])
