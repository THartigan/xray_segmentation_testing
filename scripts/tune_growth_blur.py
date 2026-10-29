"""Compare extra growth blur on the existing 20/5 split, keeping seeds fixed.

Run --phase search, then --phase native, then --phase evaluate. The previous
experiment stays in growth_tuning; this experiment writes growth_blur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from time import monotonic

import numpy as np

from mtgp import GrowthConfig
from mtgp.morphology import open_mask
from mtgp.region_growing import _grow_from_seeds, smooth_growth_image
from mtgp.tuning import config_key, dice_score, load_case, scaled_config

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'results/growth_tuning'
OUTPUT = ROOT / 'results/growth_blur'
CACHE = ROOT / 'data/.cache/growth_tuning'


def read(path: Path):
    return json.loads(path.read_text())


def save(name: str, data) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def evaluate(ids: list[str], configs: list[GrowthConfig], *, stride: int, name: str) -> list[dict]:
    """Reuse each blurred image and admission mask across spatial settings."""
    configs = list({config_key(c): c for c in configs}.values())
    def patient_scores(pid):
        case = load_case(CACHE / f'{pid}.npz', stride=stride)
        if case.seed_error:
            raise ValueError(f'{pid}: {case.seed_error}')
        last_sigma, last_key, growth_image, admitted = None, None, None, None
        records = {}
        for config in sorted(configs, key=lambda c: (c.blur_sigma, c.intensity_threshold, c.connection_radius)):
            scaled = scaled_config(config, stride)
            if scaled.blur_sigma != last_sigma:
                growth_image = smooth_growth_image(case.smoothed, case.valid, scaled.blur_sigma)
                last_sigma = scaled.blur_sigma
            key = (scaled.blur_sigma, scaled.intensity_threshold, scaled.connection_radius)
            if key != last_key:
                admitted = open_mask(case.valid & (growth_image <= scaled.intensity_threshold),
                                     radius=scaled.connection_radius)
                last_key = key
            grown = _grow_from_seeds(case.smoothed, case.valid, case.seeds, scaled,
                                     case.distances, admitted, growth_image).mask
            true_positive = int(np.count_nonzero(grown & case.reference))
            false_positive = int(np.count_nonzero(grown & ~case.reference))
            reference_area = int(np.count_nonzero(case.reference))
            records[config_key(config)] = {
                'dice': dice_score(grown, case.reference),
                'precision': true_positive / max(1, true_positive + false_positive),
                'recall': true_positive / max(1, reference_area),
                'false_positive_fraction': false_positive / max(1, reference_area),
            }
        return pid, records
    patient_rows = {}
    started = monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (pid, records) in enumerate(pool.map(patient_scores, ids), 1):
            patient_rows[pid] = records
            print(f'{name}: {i}/{len(ids)} {pid}, {monotonic()-started:.1f}s', flush=True)
    rows = []
    for config in configs:
        patients = {pid: scores[config_key(config)] for pid, scores in patient_rows.items()}
        rows.append({'config': asdict(config), 'per_patient': patients,
                     **{f'mean_{metric}': float(np.mean([p[metric] for p in patients.values()]))
                        for metric in ('dice', 'precision', 'recall', 'false_positive_fraction')}})
    rows.sort(key=lambda row: -row['mean_dice'])
    save(name, rows)
    return rows


def search(split: dict, previous: GrowthConfig) -> None:
    configs = [previous]
    for sigma, threshold, radius, distance in product(
        (0, 60, 120, 180, 240), (95, 110, 125, 140, 148.2, 160), (100, 235), (800, 1400, None),
    ):
        configs.append(replace(previous, blur_sigma=sigma, intensity_threshold=threshold,
                               connection_radius=radius, max_distance=distance))
    rows = evaluate(split['tuning_ids'], configs, stride=8, name='screening.json')
    for sigma in (0, 60, 120, 180, 240):
        best = next(row for row in rows if row['config']['blur_sigma'] == sigma)
        print('Best per blur:', best['config'], best['mean_dice'], best['mean_false_positive_fraction'], flush=True)


def native(split: dict, previous: GrowthConfig) -> None:
    screened = read(OUTPUT / 'screening.json')
    old = next(row for row in screened if GrowthConfig(**row['config']) == previous)
    # Prefer a blur candidate that reduces leakage, with Dice deciding among those.
    candidates = [row for row in screened if row['config']['blur_sigma'] > 0
                  and row['mean_false_positive_fraction'] < old['mean_false_positive_fraction']]
    if not candidates:
        raise ValueError('No blurred candidate reduced false-positive area; inspect screening results')
    candidate = GrowthConfig(**candidates[0]['config'])
    configs = [previous, candidate, replace(candidate, blur_sigma=0)]
    # Check threshold neighbours at the selected blur using every original pixel.
    configs += [replace(candidate, intensity_threshold=candidate.intensity_threshold + delta)
                for delta in (-5, 5)]
    configs += [replace(config, blur_sigma=0) for config in configs if config.blur_sigma > 0]
    rows = evaluate(split['tuning_ids'], configs, stride=1, name='native.json')
    baseline = next(row for row in rows if GrowthConfig(**row['config']) == previous)
    eligible = [row for row in rows if row['config']['blur_sigma'] > 0
                and row['mean_false_positive_fraction'] < baseline['mean_false_positive_fraction']]
    if not eligible:
        raise ValueError('Blur did not reduce native-resolution leakage; retain the previous settings')
    winner = eligible[0]
    save('best_config.json', {
        'growth_config': winner['config'], 'segmentation_config': split['segmentation_config'],
        'selection': 'Highest native mean tuning Dice among tested blurred candidates with less false-positive area than the previous winner.',
        'tuning_patient_ids': split['tuning_ids'], 'evaluation_patient_ids': split['evaluation_ids'],
        'tuning_mean_dice': winner['mean_dice'],
    })
    print('Selected:', winner['config'], winner['mean_dice'], flush=True)


def held_out(split: dict, previous: GrowthConfig) -> None:
    selected = GrowthConfig(**read(OUTPUT / 'best_config.json')['growth_config'])
    evaluate(split['evaluation_ids'], [previous, selected, replace(selected, blur_sigma=0)],
             stride=1, name='evaluation.json')
    # Evaluation never changes best_config.json.


def report(split: dict, previous: GrowthConfig) -> None:
    """Report leakage and coverage separately, including a matched blur control."""
    selection = read(OUTPUT / 'best_config.json')
    selected = GrowthConfig(**selection['growth_config'])
    training = read(OUTPUT / 'native.json')
    if (OUTPUT / 'native_ablation.json').exists():
        training += read(OUTPUT / 'native_ablation.json')
    evaluation = read(OUTPUT / 'evaluation.json')
    configs = [('Previous', previous), ('Extra blur + lower threshold', selected),
               ('Same settings, extra blur disabled', replace(selected, blur_sigma=0))]
    summary = {'growth_config': asdict(selected), 'selection': selection['selection'],
               'extra_blur_sigmas_screened': [0, 60, 120, 180, 240],
               'false_positive_fraction_definition': 'Predicted pixels outside reference / reference lung pixels, averaged equally across patients.',
               'results': {}}
    lines = [
        '# Additional blur before lung growth', '',
        'The seed pipeline stays unchanged (sigma 30). Growth receives an additional',
        f'normalized Gaussian blur with **sigma {selected.blur_sigma:g} pixels**, excluding NaNs',
        'and invalid background from both numerator and weights. Invalid pixels stay',
        'invalid. This sigma is additional, not the total; away from missing data the',
        'combined sigma is about 67 pixels. Seeds remain exactly the same.', '',
        '## Native-resolution comparison', '',
        '| Split | Settings | Dice | Precision | Recall | False-positive / reference area |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for name, rows in [('Tuning (20)', training), ('Evaluation (5)', evaluation)]:
        summary['results'][name] = {}
        for label, config in configs:
            row = next(row for row in rows if GrowthConfig(**row['config']) == config)
            summary['results'][name][label] = row
            lines.append(f"| {name} | {label} | {row['mean_dice']:.4f} | {row['mean_precision']:.4f} | {row['mean_recall']:.4f} | {row['mean_false_positive_fraction']:.4f} |")
    train = summary['results']['Tuning (20)']
    old, new = train['Previous'], train['Extra blur + lower threshold']
    reduction = 1 - new['mean_false_positive_fraction'] / old['mean_false_positive_fraction']
    lines += ['', f'The selected blurred settings reduce mean false-positive area by **{reduction:.1%}**',
              'on the tuning set relative to the previous settings, while reducing lung',
              'coverage and Dice. False-positive area means predicted pixels outside the',
              'manual lung mask divided by reference lung area; it is not a shoulder-only measure.', '',
              'The matched control keeps the new threshold and all spatial settings, but',
              'disables extra blur. This separates the effect of blur from the lower threshold.',
              'A smoother image is not automatically a better anatomical segmentation.', '',
              '## Configuration', '', '```json', json.dumps(asdict(selected), indent=2), '```', '',
              '## Method and limitations', '',
              '- Reused the same 20 tuning / 5 evaluation patients and unchanged native seed arrays.',
              '- Screened 180 configurations at stride 8, with blur sigmas 0, 60, 120, 180, 240;',
              '  thresholds 95, 110, 125, 140, 148.2, 160; connection radii 100 or 235;',
              '  maximum distances 800, 1400, or unlimited. Other controls stayed fixed.',
              '- Selected the highest screening Dice among blurred candidates with less',
              '  false-positive area than the previous configuration, then checked the',
              '  candidate and threshold neighbours at native resolution on all 20 patients.',
              '- Selected the best native Dice among those blurred candidates that reduced leakage.',
              '  This deliberately tests the requested blur approach, not an unrestricted best-Dice model.',
              '- Evaluation scores do not select parameters. These five cases have been inspected',
              '  in earlier work, so this is a repeated comparison, not a fresh untouched test set.',
              '- Seeds use no manual masks; annotations are used only for parameter scoring.', '',
              'Run `python scripts/tune_growth_blur.py --phase all` from the project root.',
              'The original experiment and settings remain in `results/growth_tuning/`.',
              'Use `blur_sigma=0` to disable extra blur. Restart the notebook kernel after',
              'updating the source to load the new configuration field.', '']
    save('summary.json', summary)
    (OUTPUT / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['search', 'native', 'evaluate', 'report', 'all'], default='all')
    args = parser.parse_args()
    split = read(PREVIOUS / 'split.json')
    previous = GrowthConfig(**read(PREVIOUS / 'best_config.json')['growth_config'])
    for name, function in [('search', search), ('native', native), ('evaluate', held_out), ('report', report)]:
        if args.phase in ('all', name):
            function(split, previous)


if __name__ == '__main__':
    main()
