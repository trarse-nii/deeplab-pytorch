from __future__ import absolute_import, division, print_function

import os
from pathlib import Path
import cv2
import numpy as np
import click

from libs.utils import evaluate

@click.group()
@click.pass_context
def main(ctx):
    """
    evaluate SS outputs
    """

    print("Mode:", ctx.invoked_subcommand)

@main.command()
@click.option(
    "-p",
    "--pred-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory Path having SS predicted outputs",
)
@click.option(
    "-l",
    "--label-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory Path having labeled outputs",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=False),
    required=True,
    help="Directory Path to save scores",
)
def evaluate_scores(pred_dir, label_dir, output_dir):
    scores_all, scores_each = evaluate(pred_dir, label_dir)
    print(scores_all)
    print(scores_each)

if __name__ == "__main__":
    main()
