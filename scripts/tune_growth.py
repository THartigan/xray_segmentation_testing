"""Tune growth on the frozen 20-patient split, then evaluate on five patients.

Run from the project root: python scripts/tune_growth.py --phase all
Intermediate results are saved for inspection and reproducibility. --phase can
resume search/native/evaluate independently. Held-out scores never select a model.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic

import numpy as np

from mtgp import GrowthConfig, SegmentationConfig, Trial
from mtgp.morphology import open_mask
from mtgp.region_growing import smooth_growth_image
from mtgp.tuning import (
    config_key, evaluate_configs, load_case, prepare_case_cache,
    refine_configs, sample_configs, score_case,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/growth_tuning'
CACHE = ROOT / 'data/.cache/growth_tuning'


def save(name, value):
    path = OUTPUT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(name):
    return json.loads((OUTPUT / name).read_text())


def unique(configs):
    return list({config_key(config): config for config in configs}.values())


def make_split():
    """Freeze 12 normal + 8 abnormal tuning; 3 normal + 2 abnormal evaluation."""
    if (OUTPUT / 'split.json').exists():
        split = read('split.json')
        if (len(split['tuning_ids']) != 20 or len(split['evaluation_ids']) != 5
                or set(split['tuning_ids']) & set(split['evaluation_ids'])):
            raise ValueError('Expected 20 tuning and 5 disjoint evaluation patients')
        return split
    trial = Trial()
    seed = 20261026
    rng = np.random.default_rng(seed)
    known = ['MCUCXR_0001_0', 'MCUCXR_0022_0', 'MCUCXR_0042_0']
    normal = [p.patient_id for p in trial if p.label == 0 and p.patient_id not in known]
    abnormal = [p.patient_id for p in trial if p.label == 1]
    rng.shuffle(normal)
    rng.shuffle(abnormal)
    split = {
        'seed': seed,
        'objective': 'Mean per-patient Dice of the union of the two lungs; seed failures score zero.',
        'tuning_ids': sorted(known + normal[:9] + abnormal[:8]),
        'evaluation_ids': sorted(normal[9:12] + abnormal[8:10]),
        'segmentation_config': asdict(SegmentationConfig()),
        'baseline_growth_config': asdict(GrowthConfig()),
        'fixed_min_seed_area': 1,
        'previously_inspected_ids_forced_into_tuning': known,
    }
    save('split.json', split)
    return split


def prepare(split):
    trial = Trial()
    ids = split['tuning_ids'] + split['evaluation_ids']
    config = SegmentationConfig(**split['segmentation_config'])
    def work(pid):
        return prepare_case_cache(trial[pid], CACHE, config)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, row in enumerate(pool.map(work, ids), 1):
            rows.append(row)
            print(f'Prepared {i}/{len(ids)}: {row}', flush=True)
    save('preprocessing.json', rows)


def search(split, workers):
    cases = [load_case(CACHE / f'{pid}.npz', stride=16) for pid in split['tuning_ids']]
    started = monotonic()
    def progress(done, total, best):
        if done % 64 == 0 or done == total:
            state = dict(phase='screening', completed=done, total=total, best_dice=best,
                         seconds=round(monotonic() - started, 1))
            save('progress.json', state)
            print(state, flush=True)
    broad = evaluate_configs(cases, sample_configs(seed=split['seed']), workers=workers, progress=progress)
    save('screening_stride16.json', broad)
    leaders = [GrowthConfig(**r['config']) for r in broad[:24]]
    local = evaluate_configs(cases, refine_configs(leaders, count=512), workers=workers, progress=progress)
    save('refinement_stride16.json', local)
    combined = sorted(broad + local, key=lambda r: -r['mean_dice'])
    finalists = unique([GrowthConfig(**r['config']) for r in combined[:96]] + [GrowthConfig()])
    del cases
    cases = [load_case(CACHE / f'{pid}.npz', stride=8) for pid in split['tuning_ids']]
    medium = evaluate_configs(cases, finalists, workers=workers, progress=progress)
    save('screening_stride8.json', medium)
    print('Reduced-resolution screening complete.', flush=True)


def native_scores(ids, configs, name):
    """Reuse exact native seed geometry and admission masks within each patient."""
    configs = unique(configs)
    def work(pid):
        case = load_case(CACHE / f'{pid}.npz')
        scores = {}
        # Group configurations so spatial-only neighbours reuse the opened mask.
        ordered = sorted(configs, key=lambda c: (c.blur_sigma, c.intensity_threshold, c.connection_radius))
        last_key, admitted, last_sigma, growth_image = None, None, None, None
        for config in ordered:
            if not case.seed_error and config.blur_sigma != last_sigma:
                growth_image = smooth_growth_image(case.smoothed, case.valid, config.blur_sigma)
                last_sigma = config.blur_sigma
            key = (config.intensity_threshold, config.connection_radius, config.blur_sigma)
            if not case.seed_error and key != last_key:
                admitted = open_mask(case.valid & (growth_image <= key[0]), radius=key[1])
                last_key = key
            scores[config_key(config)] = score_case(case, config, admissible=admitted, growth_image=growth_image)
        return pid, scores, case.seed_error
    records = []
    started = monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (pid, scores, error) in enumerate(pool.map(work, ids), 1):
            records.append({'patient_id': pid, 'scores': scores, 'seed_error': error})
            save(name.replace('.json', '_patients.json'), records)
            print(f'{name}: patient {i}/{len(ids)} {pid}; {monotonic()-started:.1f}s', flush=True)
    rows = []
    for config in configs:
        per_patient = {r['patient_id']: r['scores'][config_key(config)] for r in records}
        rows.append({'config': asdict(config), 'mean_dice': float(np.mean(list(per_patient.values()))),
                     'per_patient': per_patient})
    rows.sort(key=lambda r: (-r['mean_dice'], json.dumps(r['config'], sort_keys=True)))
    save(name, rows)
    return rows


def native(split):
    medium = read('screening_stride8.json')
    finalists = [GrowthConfig(**row['config']) for row in medium[:6]] + [GrowthConfig()]
    initial = native_scores(split['tuning_ids'], finalists, 'native_finalists.json')
    best = GrowthConfig(**initial[0]['config'])
    neighbours = [best]
    for change in (-2.0, 2.0):
        neighbours.append(replace(best, intensity_threshold=best.intensity_threshold + change))
    for change in (-10, 10):
        neighbours.append(replace(best, connection_radius=max(0, best.connection_radius + change)))
        neighbours.append(replace(best, boundary_margin=max(0, best.boundary_margin + change)))
    for change in (-100, 100):
        neighbours.append(replace(best, max_distance=max(0, (best.max_distance or 1800) + change)))
    neighbours.append(replace(best, fill_holes=not best.fill_holes))
    local = native_scores(split['tuning_ids'], neighbours, 'native_refinement.json')
    ranked = sorted(initial + local, key=lambda row: -row['mean_dice'])
    winner = ranked[0]
    save('best_config.json', {
        'growth_config': winner['config'],
        'segmentation_config': split['segmentation_config'],
        'tuning_mean_dice': winner['mean_dice'],
        'selection': 'Highest mean native-resolution Dice on the 20 tuning patients only.',
        'tuning_patient_ids': split['tuning_ids'],
        'evaluation_patient_ids': split['evaluation_ids'],
    })
    print('Native-resolution winner:', winner['config'], winner['mean_dice'], flush=True)


def evaluate(split):
    selected = read('best_config.json')
    winner = GrowthConfig(**selected['growth_config'])
    baseline = GrowthConfig(**split['baseline_growth_config'])
    # This phase never writes best_config.json or changes the chosen parameters.
    scores = native_scores(split['evaluation_ids'], [baseline, winner], 'held_out.json')
    best_row = next(r for r in scores if GrowthConfig(**r['config']) == winner)
    baseline_row = next(r for r in scores if GrowthConfig(**r['config']) == baseline)
    summary = {
        'tuning_n': 20, 'evaluation_n': 5,
        'growth_config': asdict(winner),
        'tuning_mean_dice': selected['tuning_mean_dice'],
        'evaluation_baseline_mean_dice': baseline_row['mean_dice'],
        'evaluation_tuned_mean_dice': best_row['mean_dice'],
        'evaluation_change': best_row['mean_dice'] - baseline_row['mean_dice'],
        'per_patient_baseline': baseline_row['per_patient'],
        'per_patient_tuned': best_row['per_patient'],
        'note': 'Mean patient Dice; missing seed pairs score zero. Five held-out cases give an imprecise generalization estimate. Best found in this search, not a proof of a global optimum.',
    }
    training = read('native_finalists.json') + read('native_refinement.json')
    train_baseline = next(r for r in training if GrowthConfig(**r['config']) == baseline)
    train_winner = next(r for r in training if GrowthConfig(**r['config']) == winner)
    summary['tuning_baseline_mean_dice'] = train_baseline['mean_dice']
    failures = {r['patient_id']: r['seed_error'] for r in read('preprocessing.json') if r['seed_error']}
    summary['seed_failures'] = failures
    recovered = {r['patient_id']: r['seed_sources'] for r in read('preprocessing.json')
                 if any(source.startswith('fallback:') for source in r['seed_sources'])}
    summary['recovered_seeds'] = recovered
    summary['evaluation_scope'] = (
        'Held out from growth-parameter selection. A missing seed in an evaluation image '
        'informed annotation-free seed recovery; no evaluation Dice selected that change.'
    )
    seen = set()
    for name in ['screening_stride16.json', 'refinement_stride16.json',
                 'screening_stride8.json', 'native_finalists.json', 'native_refinement.json']:
        seen.update(json.dumps(row['config'], sort_keys=True) for row in read(name))
    summary['unique_configurations_tested'] = len(seen)
    save('summary.json', summary)
    with (OUTPUT / 'patient_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'patient_id', 'baseline_dice', 'tuned_dice', 'change', 'seed_error'])
        writer.writeheader()
        for group, old, new in [('tuning', train_baseline, train_winner), ('evaluation', baseline_row, best_row)]:
            for pid, score in new['per_patient'].items():
                previous = old['per_patient'][pid]
                writer.writerow(dict(split=group, patient_id=pid, baseline_dice=previous,
                                     tuned_dice=score, change=score-previous,
                                     seed_error=failures.get(pid, '')))
    report = f'''# Growth-parameter tuning

One global configuration was selected using **20 tuning patients** (12 normal,
8 abnormal). The **5 evaluation patients** (3 normal, 2 abnormal) were held out
from growth-parameter selection. The three examples inspected in earlier experiments were
forced into the tuning set. Split seed: {split['seed']}.

## Native-resolution results

| Split | Patients | Baseline mean Dice | Tuned mean Dice | Change |
|---|---:|---:|---:|---:|
| Tuning | 20 | {train_baseline['mean_dice']:.6f} | {train_winner['mean_dice']:.6f} | {train_winner['mean_dice']-train_baseline['mean_dice']:+.6f} |
| Held-out evaluation | 5 | {baseline_row['mean_dice']:.6f} | {best_row['mean_dice']:.6f} | {best_row['mean_dice']-baseline_row['mean_dice']:+.6f} |

Dice is computed on the union of the two lungs against the union of the supplied
manual masks, using every native-resolution pixel. Each patient has equal weight.
Missing seed pairs score zero and remain in the denominator.

## Selected configuration

```json
{json.dumps(asdict(winner), indent=2)}
```

## Search procedure

- {len(seen)} distinct configurations tested, including the original baseline.
- Initial Sobol screening of 2,048 proposals on all 20 tuning cases at stride 16,
  plus the baseline; 512 local proposals around the leading candidates.
- Recheck the best 96 screening entries plus the baseline at stride 8.
- Compare six finalists and the baseline at full resolution, then test local
  parameter neighbours at full resolution. Select by tuning Dice only.
- Keep Gaussian smoothing, initial threshold/morphology, and nearest-blob seed
  selection with missing-side recovery fixed. Spatial lengths are scaled during screening
  and reported in original pixels. No evaluation scores select parameters.

The broad search covers thresholds 70–180, connection radii 0–300 pixels,
maximum distances about 150–2,000 pixels or unlimited, margins 0–200 pixels,
and both hole-filling settings. Local searches can extend these ranges.
The result is the best found within this search, not a proven global optimum.
Five evaluation cases give only a small-sample estimate of generalization.

## Seed failures

'''
    report += '\n'.join(f'- `{pid}`: {error}' for pid, error in failures.items()) or 'None.'
    report += '''

## Seed verification and evaluation scope

The first 40 Trial patients and the previously failing `MCUCXR_0309_1` all have
two nonempty, connected seeds, one on each image side, verified at native
resolution. See `seed_verification.json` and `seeds_first40_page1.png` /
`seeds_first40_page2.png`. This checks seed coverage, not perfect anatomical masks.

Missing-side recovery was added after observing a missing seed in an evaluation
image. It uses image intensities and morphology only, without manual masks or
evaluation Dice. Existing pairs are preserved; all 20 tuning seed arrays were
verified bit-for-bit unchanged, so the completed screening remains applicable.
The five evaluation cases were held out from growth-parameter selection, but
were not completely untouched during seed development. Both baseline and tuned
growth use the same improved seeds.

Recovered seeds:

'''
    report += '\n'.join(f'- `{pid}`: {sources}' for pid, sources in recovered.items()) or 'None.'
    report += '''

See `patient_scores.csv` for every individual result,
`split.json` for patient IDs, and the screening/native JSON files for all scores.

## Reproduce

From the project root, run `python scripts/tune_growth.py --phase all`.
Use `--phase prepare`, `search`, `native`, or `evaluate` to run an individual stage.
The exact preprocessing cache is stored under `data/.cache/growth_tuning/`
(ignored by Git, approximately 1.2 GB for this run); deleting it only requires
rerunning preparation. The search does not overwrite the library defaults.
'''
    (OUTPUT / 'report.md').write_text(report)
    plot_scores(train_baseline, train_winner, baseline_row, best_row, split)
    save('progress.json', {'phase': 'complete', 'tuning_n': 20, 'evaluation_n': 5,
                           'summary': 'summary.json', 'configuration': 'best_config.json'})
    print(json.dumps(summary, indent=2), flush=True)


def plot_scores(train_baseline, train_winner, baseline_row, best_row, split):
    """Render stored results without recomputing predictions or changing selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    x = np.arange(2)
    old = [train_baseline['mean_dice'], baseline_row['mean_dice']]
    new = [train_winner['mean_dice'], best_row['mean_dice']]
    axes[0].bar(x-.18, old, .36, label='Baseline', color='#607d8b')
    axes[0].bar(x+.18, new, .36, label='Tuned', color='#168c83')
    axes[0].set_xticks(x, ['Tuning (20)', 'Held out (5)'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Mean patient Dice')
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2)
    for i, (before, after) in enumerate(zip(old, new)):
        axes[0].text(i-.18, before+.015, f'{before:.3f}', ha='center')
        axes[0].text(i+.18, after+.015, f'{after:.3f}', ha='center')
    for i, pid in enumerate(split['evaluation_ids']):
        before, after = baseline_row['per_patient'][pid], best_row['per_patient'][pid]
        axes[1].plot([before, after], [i, i], color='#888888')
        axes[1].scatter(before, i, color='#607d8b')
        axes[1].scatter(after, i, color='#168c83')
    axes[1].set_yticks(range(5), split['evaluation_ids'])
    axes[1].set_xlim(-.02, 1)
    axes[1].set_xlabel('Dice on each held-out patient')
    axes[1].set_title('Individual evaluation results')
    fig.savefig(OUTPUT / 'scores.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'prepare', 'search', 'native', 'evaluate'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split = make_split()
    for name, function in [('prepare', prepare), ('search', lambda s: search(s, args.workers)),
                           ('native', native), ('evaluate', evaluate)]:
        if args.phase in ('all', name):
            function(split)


if __name__ == '__main__':
    main()
