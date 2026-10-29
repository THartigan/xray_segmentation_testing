"""Tune marker competition on 20 patients, then compare on the existing five.

The native preprocessing/seed cache is shared with the original experiment.
Screening scores use the working grid; final selection and evaluation score
restored masks against all original-resolution reference pixels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from time import monotonic

import numpy as np

from mtgp.competitive_growing import (
    CompetitiveGrowthConfig, _grow_competitive_from_seeds, _working_prediction, _restore_prediction,
)
from mtgp.seeding import LungSeeds
from mtgp.tuning import load_case, dice_score

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'results/growth_tuning'
OUTPUT = ROOT / 'results/growth_methods'
CACHE = ROOT / 'data/.cache/growth_tuning'


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def metrics(mask, reference):
    tp = int(np.count_nonzero(mask & reference))
    fp = int(np.count_nonzero(mask & ~reference))
    area = max(1, int(np.count_nonzero(reference)))
    return dict(dice=dice_score(mask, reference), precision=tp/max(1, tp+fp), recall=tp/area,
                false_positive_fraction=fp/area)


def evaluate(ids, configs, *, screening, name):
    configs = list({json.dumps(asdict(c), sort_keys=True): c for c in configs}.values())
    def work(pid):
        case = load_case(CACHE / f'{pid}.npz', cache_distances=False)
        if case.seed_error:
            raise ValueError(f'{pid}: {case.seed_error}')
        scores = []
        for config in configs:
            if screening:
                stride = max(1, int(np.ceil(max(case.valid.shape) / config.working_max_size)))
                s = np.s_[::stride, ::stride]
                seeds = LungSeeds(case.seeds.labels[s], tuple(c/stride for c in case.seeds.chest_center),
                                  case.seeds.component_ids, case.seeds.sources)
                working, _, _ = _working_prediction(case.smoothed[s], case.valid[s], seeds, config, stride)
                mask = _restore_prediction(working, case.valid[s], seeds, 1) > 0
                reference = case.reference[s]
            else:
                result = _grow_competitive_from_seeds(case.smoothed, case.valid, case.seeds, config)
                mask, reference = result.mask, case.reference
            scores.append(metrics(mask, reference))
        return pid, scores
    records = {}
    start = monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, (pid, scores) in enumerate(pool.map(work, ids), 1):
            records[pid] = scores
            print(f'{name}: {index}/{len(ids)} {pid}; {monotonic()-start:.1f}s', flush=True)
    rows = []
    for i, config in enumerate(configs):
        patients = {pid: values[i] for pid, values in records.items()}
        rows.append({'config': asdict(config), 'per_patient': patients,
                     **{f'mean_{metric}':float(np.mean([p[metric] for p in patients.values()]))
                        for metric in ('dice', 'precision', 'recall', 'false_positive_fraction')}})
    rows.sort(key=lambda r: -r['mean_dice'])
    save(name, rows)
    return rows


def search(split):
    configs = []
    for method, threshold, sigma, margin in product(
        ('random_walker', 'watershed'), (130,160,190), (0,30,60), (100,265),
    ):
        for beta in ((30,130,500,2000) if method == 'random_walker' else (130,)):
            configs.append(CompetitiveGrowthConfig(method=method, background_threshold=threshold,
                                                  blur_sigma=sigma, boundary_margin=margin, beta=beta))
    rows = evaluate(split['tuning_ids'], configs, screening=True, name='screening.json')
    for method in ('random_walker', 'watershed'):
        best = next(r for r in rows if r['config']['method'] == method)
        print('Screening leader:', best['config'], best['mean_dice'], flush=True)


def native(split):
    screened = read(OUTPUT / 'screening.json')
    configs = []
    for method in ('random_walker', 'watershed'):
        configs.extend(CompetitiveGrowthConfig(**r['config']) for r in
                       [r for r in screened if r['config']['method'] == method][:3])
    rows = evaluate(split['tuning_ids'], configs, screening=False, name='native.json')
    winner = rows[0]
    save('best_config.json', {'competitive_config': winner['config'],
                             'segmentation_config': split['segmentation_config'],
                             'tuning_ids': split['tuning_ids'], 'evaluation_ids': split['evaluation_ids'],
                             'tuning_mean_dice': winner['mean_dice'],
                             'selection': 'Highest mean native-resolution Dice on 20 tuning patients among six screening finalists.'})
    print('Native winner:', winner['config'], winner['mean_dice'], flush=True)


def held_out(split):
    rows = read(OUTPUT / 'native.json')
    configs = [CompetitiveGrowthConfig(**next(r for r in rows if r['config']['method']==method)['config'])
               for method in ('random_walker', 'watershed')]
    evaluate(split['evaluation_ids'], configs, screening=False, name='evaluation.json')


def report(split):
    """Summarize the matched experiment without using evaluation for selection."""
    training = read(OUTPUT / 'native.json')
    evaluation = read(OUTPUT / 'evaluation.json')
    blur = read(ROOT / 'results/growth_blur/summary.json')['results']
    selected = read(OUTPUT / 'best_config.json')
    results = {}
    for group, rows, blur_key in [('tuning',training,'Tuning (20)'),('evaluation',evaluation,'Evaluation (5)')]:
        results[group] = {
            'Previous threshold': blur[blur_key]['Previous'],
            'Threshold + extra blur': blur[blur_key]['Extra blur + lower threshold'],
            'Lower threshold without extra blur': blur[blur_key]['Same settings, extra blur disabled'],
        }
        for method in ('random_walker','watershed'):
            config = next(r for r in training if r['config']['method'] == method)['config']
            results[group][method] = next(r for r in rows if r['config'] == config)
    old, new = results['evaluation']['Previous threshold'], results['evaluation']['watershed']
    reduction = 1 - new['mean_false_positive_fraction'] / old['mean_false_positive_fraction']
    improved = sum(new['per_patient'][pid]['dice'] > old['per_patient'][pid]['dice']
                   for pid in split['evaluation_ids'])
    save('summary.json', {'results':results, 'recommended_config':selected['competitive_config'],
                          'evaluation_false_positive_reduction':reduction,
                          'evaluation_patients_improved':improved,
                          'evaluation_scope':'Repeated five-patient comparison, not an untouched test set. Parameters and method selected on 20 tuning patients.'})
    lines = ['# Growth-method comparison', '',
             '**Recommendation: marker-controlled watershed with the existing lung seeds.**', '',
             'The tested watershed implementation improves both mean Dice and leakage compared',
             'with threshold propagation. Stronger blur alone does not solve overgrowth.', '',
             '## Original-resolution scores', '',
             '| Method | Tuning Dice (20) | Evaluation Dice (5) | Evaluation precision | Evaluation recall | Evaluation FP/reference area |',
             '|---|---:|---:|---:|---:|---:|']
    for method in results['tuning']:
        a, b = results['tuning'][method], results['evaluation'][method]
        lines.append(f"| {method.replace('_',' ')} | {a['mean_dice']:.4f} | {b['mean_dice']:.4f} | {b['mean_precision']:.4f} | {b['mean_recall']:.4f} | {b['mean_false_positive_fraction']:.4f} |")
    lines += ['', f'Watershed reduces mean false-positive/reference area by **{reduction:.1%}** on',
              f'the five evaluation patients. Dice improves on **{improved}/5**. This metric counts all',
              'predicted pixels outside the supplied lung masks, not only shoulder leakage.',
              'The remaining recall deficit shows that parts of the lungs are still missed.', '',
              '## Why change the method?', '',
              '- Threshold propagation accepts every reachable pixel below one intensity cutoff.',
              '  Once a low-intensity path reaches the shoulder, there is no competing background label.',
              '- Marker-controlled watershed uses image gradients as barriers and floods from lung',
              '  and background markers. The background can claim shoulder tissue before the lungs do.',
              '  See the [official watershed example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html).',
              '- Random walker also uses competing markers, with diffusion discouraged across strong',
              '  gradients. With these automatic markers it under-segments more severely; its lower',
              '  Dice does not justify switching to it here. See the [official random-walker example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_random_walker_segmentation.html).',
              '- Active contours are another possible next experiment, but were not benchmarked in',
              '  this run. There is no evidence here that they outperform the tested watershed.', '',
              '## Selected watershed settings', '', '```json',
              json.dumps(selected['competitive_config'],indent=2), '```', '',
              '`beta` is used only by random walker. Extra blur is disabled for the selected watershed;',
              'the original normalized sigma-30 seed preprocessing remains in effect.', '',
              '## Implementation and fair comparison', '',
              '- Both alternative methods use exactly the existing native lung seeds.',
              '- Automatic background markers: valid perimeter/margin, intensities above the',
              '  background threshold, and the chest-centre column. Seed labels take precedence.',
              '  No reference mask enters marker construction or inference.',
              '- The longest working-grid side is at most 384 pixels. Predictions are restored by',
              '  nearest-neighbour sampling; original seeds are restored exactly, new invalid pixels',
              '  are excluded, and disconnected additions are discarded. Native output shape is preserved.',
              '- This working resolution is part of the method, not just a display reduction. It bounds',
              '  graph memory and introduces small boundary quantization. All final Dice and leakage',
              '  measurements compare restored masks with every original-resolution reference pixel.',
              '- Screened 90 configurations on the same 20 tuning patients: 72 random-walker and',
              '  18 watershed settings. Background thresholds 130/160/190, additional sigma 0/30/60,',
              '  margins 100/265; random-walker beta 30/130/500/2000. No alternative seed tuning.',
              '- Rechecked the top three settings per method at original output resolution. The method',
              '  and configuration were chosen by mean tuning Dice, then frozen before evaluation.',
              '- The same five evaluation images were inspected during earlier experiments and prompted',
              '  changes. Their scores are a repeated comparison, not an unbiased fresh test estimate.', '',
              '## Per-patient evaluation Dice', '',
              '| Patient | Previous threshold | Watershed | Change |', '|---|---:|---:|---:|']
    for pid in split['evaluation_ids']:
        before, after = old['per_patient'][pid]['dice'], new['per_patient'][pid]['dice']
        lines.append(f'| {pid} | {before:.4f} | {after:.4f} | {after-before:+.4f} |')
    lines += ['', '## Reproduce', '',
              '`python scripts/compare_growth_methods.py --phase all` uses the cached native',
              'preprocessing from `data/.cache/growth_tuning/`. Regenerate that cache with',
              '`python scripts/tune_growth.py --phase prepare` if needed. The blur-comparison',
              'summary is also read for the historical controls.', '',
              'The notebook retains the existing sigma-5, first-40 baseline experiment and adds',
              'separate blur and recommended watershed cells using saved sigma-30 settings.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines))


def plot_comparison(split):
    """Render all five cases, and assert predictions reproduce saved scores."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mtgp import Trial, GrowthConfig
    from mtgp.region_growing import _grow_from_seeds

    previous = GrowthConfig(**read(PREVIOUS / 'best_config.json')['growth_config'])
    selected = CompetitiveGrowthConfig(**read(OUTPUT / 'best_config.json')['competitive_config'])
    scores = next(row for row in read(OUTPUT / 'evaluation.json') if row['config'] == asdict(selected))
    trial = Trial()
    previews = []
    for pid in split['evaluation_ids']:
        case = load_case(CACHE / f'{pid}.npz', cache_distances=False)
        old = _grow_from_seeds(case.smoothed, case.valid, case.seeds, previous).mask
        new = _grow_competitive_from_seeds(case.smoothed, case.valid, case.seeds, selected).mask
        old_score, new_score = dice_score(old, case.reference), dice_score(new, case.reference)
        if new_score != scores['per_patient'][pid]['dice']:
            raise AssertionError(f'{pid}: prediction no longer matches the recorded score')
        step = max(1, int(np.ceil(max(case.valid.shape) / 650)))
        s = np.s_[::step, ::step]
        previews.append((pid, trial[pid].image[s].copy(), old[s].copy(), new[s].copy(),
                         case.reference[s].copy(), old_score, new_score))

    def render(rows, filename):
        fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.6 * len(rows)),
                                 squeeze=False, layout='constrained')
        fig.suptitle('Cyan = predicted lung; orange = manual reference', fontsize=12)
        for row, (pid, image, old, new, reference, old_score, new_score) in enumerate(rows):
            for column, (name, mask, score) in enumerate((('Previous threshold',old,old_score),
                                                         ('Marker-controlled watershed',new,new_score))):
                ax = axes[row,column]
                ax.imshow(image,cmap='gray')
                ax.contour(reference,levels=[.5],colors=['orange'],linewidths=.9)
                ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=.9)
                ax.set_title(f'{pid} — {name}\nDice {score:.3f}',fontsize=10)
                ax.axis('off')
        fig.savefig(OUTPUT / filename,dpi=140)
        plt.close(fig)
    render(previews,'evaluation_comparison.png')
    render([row for row in previews if row[0]=='MCUCXR_0309_1'],'difficult_case_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all','search','native','evaluate','report','plot'], default='all')
    args = parser.parse_args()
    split = read(PREVIOUS / 'split.json')
    for phase, function in [('search',search),('native',native),('evaluate',held_out),('report',report),('plot',plot_comparison)]:
        if args.phase in ('all',phase):
            function(split)


if __name__ == '__main__':
    main()
