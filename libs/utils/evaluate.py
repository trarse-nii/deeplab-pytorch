from __future__ import absolute_import, division, print_function

import os
from pathlib import Path
import cv2
import numpy as np

from metric import scores_with_labels

def evaluate(pred_dir, label_dir):
    pred_paths = list(Path(pred_dir).glob("*.png"))

    pred_imgs, label_imgs = [], []
    labels_true = []
    scores_each = []

    for pred_path in pred_paths:
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        label_img  = cv2.imread(str(label_dir)+'/'+os.path.basename(pred_path), cv2.IMREAD_GRAYSCALE)

        pred_img = cv2.resize(pred_img, (label_img.shape[1],label_img.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)

        pred_img_ex = np.expand_dims(pred_img, 0)
        label_img_ex = np.expand_dims(label_img, 0)

        pred_imgs += list(pred_img_ex)
        label_imgs += list(label_img_ex)
        
        labels = np.unique(label_img)
        labels_true = np.append(labels_true, label_img)

        pred_scores = scores_with_labels(label_img_ex, pred_img_ex, labels)
        pred_scores["name"] = os.path.basename(pred_path)
        scores_each += list([pred_scores.items()])

    labels = np.unique(labels_true)
    scores_all = scores_with_labels(label_imgs, pred_imgs, labels)

    return scores_all, scores_each