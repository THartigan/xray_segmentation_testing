"""Render matched growth comparisons using fresh full-resolution inference."""
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from mtgp import GrowthConfig, SegmentationConfig, Trial, grow_lungs, segment_lungs
from mtgp.tuning import dice_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/growth_blur'


def main():
    previous = json.loads((ROOT / 'results/growth_tuning/best_config.json').read_text())
    current = json.loads((OUTPUT / 'best_config.json').read_text())
    config = GrowthConfig(**current['growth_config'])
    configs = [GrowthConfig(**previous['growth_config']), config, replace(config, blur_sigma=0)]
    names = ['Previous growth', 'Lower threshold + extra blur', 'Same lower threshold, no extra blur']
    trial = Trial()
    patients = [trial['MCUCXR_0001_0'], trial['MCUCXR_0309_1']]
    fig, axes = plt.subplots(2, 4, figsize=(15, 11), layout='constrained')
    fig.suptitle('Growth blur comparison: cyan = predicted; orange = reference', fontsize=14)
    for row, patient in enumerate(patients):
        image = patient.image
        seg = segment_lungs(image, SegmentationConfig(**current['segmentation_config']))
        reference = patient.lung_mask
        step = max(1, int(np.ceil(max(image.shape) / 650)))
        sample = np.s_[::step, ::step]
        first_seeds = None
        for column, (setting, name) in enumerate(zip(configs, names), 1):
            grown = grow_lungs(seg, setting)
            if first_seeds is None:
                first_seeds = grown.seeds.labels.copy()
            else:
                np.testing.assert_array_equal(first_seeds, grown.seeds.labels)
            score = dice_score(grown.mask, reference)
            ax = axes[row, column]
            ax.imshow(image[sample], cmap='gray')
            ax.contour(reference[sample], levels=[.5], colors=['orange'], linewidths=.8)
            ax.contour(grown.mask[sample], levels=[.5], colors=['cyan'], linewidths=.8)
            ax.set_title(f'{name}\nDice {score:.3f}', fontsize=10)
            if column == 2:
                axes[row, 0].imshow(grown.growth_image[sample], cmap='gray')
                axes[row, 0].set_title(f'{patient.patient_id}\nGrowth image: extra σ={setting.blur_sigma:g}', fontsize=10)
            print(patient.patient_id, name, score, flush=True)
        for ax in axes[row]:
            ax.axis('off')
    fig.savefig(OUTPUT / 'comparison.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
