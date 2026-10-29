"""Verify native-resolution seed coverage and render an annotation-free audit.

Run from the project root: python scripts/verify_seeds.py
This checks the first 40 Trial patients and the previously failing MCUCXR_0309_1.
Manual lung masks are not read or used by this verification.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

from mtgp import Trial, segment_lungs, find_lung_seeds

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/growth_tuning'


def inspect_patient(patient):
    image = patient.image
    result = segment_lungs(image)
    seeds = find_lung_seeds(result)
    counts, centroids = [], []
    for side in (1, 2):
        mask = seeds.labels == side
        count = int(mask.sum())
        assert count > 0, f'{patient.patient_id}: missing side {side}'
        assert label(mask, connectivity=2).max() == 1, f'{patient.patient_id}: disconnected seed'
        centroid_x = float(np.dot(np.arange(mask.shape[1]), mask.sum(axis=0)) / count)
        assert (centroid_x < seeds.chest_center[1]) == (side == 1)
        counts.append(count)
        centroids.append(centroid_x)
    row = dict(patient_id=patient.patient_id, left_pixels=counts[0], right_pixels=counts[1],
               chest_center=list(seeds.chest_center), seed_centroid_columns=centroids,
               sources=list(seeds.sources), verified=True)
    stride = max(1, int(np.ceil(max(image.shape) / 450)))
    return row, image[::stride, ::stride].copy(), seeds.labels[::stride, ::stride].copy()


def main():
    trial = Trial()
    patients = list(trial[:40]) + [trial['MCUCXR_0309_1']]
    records, previews = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (row, image, seeds) in enumerate(pool.map(inspect_patient, patients), 1):
            records.append(row)
            previews.append((image, seeds))
            print(f"Verified {i}/{len(patients)} {row['patient_id']}: {row['sources']}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'seed_verification.json').write_text(json.dumps({
        'first_40_passed': all(r['verified'] for r in records[:40]),
        'previously_failing_case_passed': records[-1]['verified'],
        'checks': 'Nonempty, single 8-connected seed on each image side; no manual annotations used.',
        'patients': records,
    }, indent=2) + '\n')
    for page, start in enumerate((0, 20), 1):
        fig, axes = plt.subplots(4, 5, figsize=(16, 15), layout='constrained')
        fig.suptitle(f'Native-resolution seed audit: patients {start+1}–{start+20}\nCyan: image-left; orange: image-right')
        for ax, record, (image, seeds) in zip(axes.flat, records[start:start+20], previews[start:start+20]):
            ax.imshow(image, cmap='gray')
            for side, color in ((1, 'cyan'), (2, 'orange')):
                ax.contour(seeds == side, levels=[.5], colors=[color], linewidths=.8)
            ax.set_title(record['patient_id'], fontsize=9)
            ax.axis('off')
        fig.savefig(OUTPUT / f'seeds_first40_page{page}.png', dpi=120)
        plt.close(fig)
    print('All first 40 patients and the previously failing case passed.', flush=True)


if __name__ == '__main__':
    main()
