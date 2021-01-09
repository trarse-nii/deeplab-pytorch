from __future__ import absolute_import, division, print_function

import os
from pathlib import Path
import cv2
import numpy as np
import click

from libs.utils import evaluate_from_dir

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
@click.option(
    "-s",
    "--suffix",
    type=click.Path(exists=False),
    required=True,
    help="suffix of output file",
)
def evaluate_scores(pred_dir, label_dir, output_dir, suffix):
    scores_all, scores_each = evaluate_from_dir(pred_dir, label_dir)
    
    # debug描画用
    #print(scores_all)
    #print(scores_each)

    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)

    f_whole = open(output_dir + '/' + 'eval_whole_' + suffix + '.txt', 'w')
    f_each = open(output_dir + '/' + 'eval_each_' + suffix + '.csv', 'w')

    # クラス別スコアをカラム別で記録
    class_iou_unique = scores_all['Class IoU']
    max_class_num = np.array([int(n) for n in class_iou_unique.keys()]).max()
    class_list = np.arange(max_class_num + 1)
    class_list_str = ",".join(map(str, class_list))

    # カラムタイトル
    column_title = 'ImgName,MeanAccuracy,FrequencyWeightedIoU,MeanIoU,' + class_list_str
    f_each.write(f"{column_title}\n")

    # 値書き込み
    for score in scores_each:
        # クラス別IoU値の配列初期化
        class_value_list = ["" for _ in range(max_class_num + 1)]

        print('name: ' + score['name'])
        line_str = f"{score['name']},{score['Mean Accuracy']},{score['Frequency Weighted IoU']},{score['Mean IoU']},"

        for key, value in score['Class IoU'].items():
            class_value_list[int(key)] = value

        class_value_str = ",".join(map(str,class_value_list))
        line_str += class_value_str

        f_each.write(f"{line_str}\n")

    for key, value in scores_all.items():
        f_whole.write(f'{key}: {value}\n')
    
    f_whole.close()
    f_each.close()

if __name__ == "__main__":
    main()
