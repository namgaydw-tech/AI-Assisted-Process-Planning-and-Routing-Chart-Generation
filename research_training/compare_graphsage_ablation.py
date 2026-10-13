"""Compare real-edge GraphSAGE against no-edge GraphSAGE on frozen VALIDATION."""
from pathlib import Path
import json

base = Path('research_outputs')
a = base / 'fixed_benchmark_graphsage_v1'
b = base / 'fixed_benchmark_graphsage_no_edges_v1'


def read(folder: Path, name: str):
    p = folder / name
    if not p.is_file():
        raise SystemExit(f'Missing file: {p}. Complete both training runs first.')
    return json.loads(p.read_text(encoding='utf-8'))


real = read(a, 'validation_metrics.json')
zero = read(b, 'validation_metrics.json')
if zero.get('adjacency_mode') != 'all_edges_removed_train_and_validation':
    raise SystemExit('No-edge run lacks the verified ablation mode flag.')
for key in ('manifest_sha256', 'input_features', 'input_dim', 'hidden_dim',
            'seed', 'training_parts', 'training_faces', 'n_parts', 'n_faces'):
    if real.get(key) != zero.get(key):
        raise SystemExit(f'Comparison not controlled: mismatch in {key}')

print('\nFORGEPATH GRAPHSAGE ADJACENCY ABLATION (VALIDATION ONLY)')
print(f"{'Metric':<23} {'Real edges':>12} {'No edges':>12} {'Delta (pp)':>12}")
for metric in ('validation_accuracy', 'validation_balanced_accuracy',
               'validation_macro_f1', 'validation_weighted_f1'):
    x, y = float(real[metric]), float(zero[metric])
    print(f'{metric:<23} {x * 100:>11.3f}% {y * 100:>11.3f}% {(x-y)*100:>+11.3f}')

per_real = read(a, 'validation_per_class.json')
per_zero = read(b, 'validation_per_class.json')
keys = [k for k, v in per_real.items() if isinstance(v, dict)
        and k not in ('macro avg', 'weighted avg', 'micro avg', 'samples avg')
        and k in per_zero and 'f1-score' in v and 'f1-score' in per_zero[k]]
rows = []
for k in keys:
    rf, zf = per_real[k]['f1-score'], per_zero[k]['f1-score']
    rows.append((k, rf, zf, (rf - zf) * 100))
print('\nPER-CLASS VALIDATION F1 DIFFERENCE (real-edge minus no-edge)')
for k, rf, zf, delta in sorted(rows, key=lambda r: -r[3]):
    print(f'{k:<32} {rf * 100:>7.2f}% {zf * 100:>7.2f}% {delta:>+8.2f} pp')
print('\nReserved TEST split was not accessed. Do not report this as final holdout performance.')
