"""ForgePath Phase 6: controlled NO-ADJACENCY GraphSAGE ablation.

This is based on the exact Phase 5 training source. Its model, preprocessing,
optimizer, seed and train/validation split are UNCHANGED. The ONLY intended
experimental manipulation is that every graph has zero edges in both training
and validation, so there is no neighbor message passing. Saved artifacts have
an entirely separate directory; the Phase 5 checkpoint is never overwritten.

The frozen final TEST split is never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

N_CLASSES = 16
FEATURE_CLASSES = [
    'rectangular_through_slot', 'triangular_through_slot', 'rectangular_passage',
    'triangular_passage', '6sides_passage', 'rectangular_through_step',
    '2sides_through_step', 'slanted_through_step', 'rectangular_blind_step',
    'triangular_blind_step', 'rectangular_blind_slot', 'rectangular_pocket',
    'triangular_pocket', '6sides_pocket', 'chamfer', 'stock',
]


class SageLayer(nn.Module):
    """Mean-neighbor aggregation, independent of torch_geometric."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        neigh_sum = torch.zeros_like(x)
        count = x.new_zeros((x.shape[0], 1))
        if src.numel():
            neigh_sum.index_add_(0, dst, x.index_select(0, src))
            count.index_add_(0, dst, x.new_ones((src.numel(), 1)))
        neigh_mean = neigh_sum / count.clamp(min=1.0)
        return self.linear(torch.cat((x, neigh_mean), dim=1))


class FaceGraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layer1 = SageLayer(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = SageLayer(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, N_CLASSES)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.norm1(self.layer1(features, edge_index))))
        x = self.dropout(torch.relu(self.norm2(self.layer2(x, edge_index))))
        return self.classifier(x)


@dataclass
class Graph:
    x: torch.Tensor
    y: torch.Tensor
    edges: torch.Tensor


class GraphDataset(Dataset):
    def __init__(self, graphs: list[Graph]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> Graph:
        return self.graphs[i]


def collate_graphs(items: list[Graph]):
    x = torch.cat([g.x for g in items], dim=0)
    y = torch.cat([g.y for g in items], dim=0)
    offset = 0
    edges = []
    for g in items:
        edges.append(g.edges + offset)
        offset += len(g.y)
    e = torch.cat(edges, dim=1) if edges else torch.empty((2, 0), dtype=torch.int64)
    return x, y, e


def load_graphs(frame: pd.DataFrame, transformed: np.ndarray, graph_dir: Path) -> list[Graph]:
    records = []
    # Sorting and using CSV row indices is critical: graph nodes are indexed by STEP face ID.
    for part, group in frame.groupby('part_id', sort=True):
        face_ids = group['face_id'].to_numpy(dtype=np.int64)
        path = graph_dir / f'{part}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing graph for {part}: {path}')
        with np.load(path, allow_pickle=False) as graph:
            graph_fids = graph['face_ids'].astype(np.int64)
            edge_index = graph['edge_index'].astype(np.int64)
        if not np.array_equal(face_ids, graph_fids):
            raise ValueError(f'Face ID mismatch for {part}')
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f'Bad graph adjacency array for {part}')
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(group)):
            raise ValueError(f'Out-of-range graph edge for {part}')
        records.append(Graph(
            x=torch.from_numpy(np.ascontiguousarray(transformed[group.index.to_numpy()], dtype=np.float32)),
            y=torch.from_numpy(group['label'].to_numpy(dtype=np.int64).copy()),
            edges=torch.from_numpy(edge_index.copy()),
        ))
    return records


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    actual, predicted = [], []
    for features, labels, edges in loader:
        logits = model(features.to(device), edges.to(device))
        actual.append(labels.numpy())
        predicted.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(actual)
    p = np.concatenate(predicted)
    score = float(f1_score(y, p, average='macro', labels=list(range(N_CLASSES)), zero_division=0))
    return score, y, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('research_outputs/mfcad_full_face_features.csv'))
    parser.add_argument('--manifest', type=Path, default=Path('research_outputs/splits/mfcad_full_provisional_group_v1.json'))
    parser.add_argument('--graphs', type=Path, default=Path('research_outputs/graphs/mfcad_adjacency'))
    parser.add_argument('--out', type=Path, default=Path('research_outputs/fixed_benchmark_graphsage_no_edges_v1'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch-parts', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'Output folder exists and is non-empty: {args.out}')
    if args.epochs < 1 or args.patience < 1 or args.batch_parts < 1:
        raise SystemExit('epochs, patience and batch-parts must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (
        'cpu' if args.device == 'auto' else args.device)
    if dev == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but PyTorch cannot access CUDA')
    device = torch.device(dev)

    raw_manifest = args.manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    splits = {k: set(v) for k,v in manifest['splits'].items()}
    if set(splits) != {'train','validation','test'}:
        raise ValueError('Invalid manifest')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if splits[a] & splits[b]:
            raise ValueError(f'Part-ID overlap between {a} and {b}')

    audit_file = args.graphs.parent/'mfcad_adjacency.audit.json'
    if not audit_file.exists():
        raise SystemExit('Graph extraction audit missing; export all CAD graphs first')
    audit = json.loads(audit_file.read_text(encoding='utf-8'))
    total_parts = sum(map(len,splits.values()))
    if audit['requested'] != total_parts or audit['graphs_written'] != total_parts or audit['skipped']:
        raise SystemExit('Graph audit is not complete for all manifest parts')
    df = pd.read_csv(args.csv)
    if len(df) != audit['faces']:
        raise ValueError('Extracted CSV faces disagree with graph audit')
    if set(df.part_id.unique()) != set.union(*splits.values()):
        raise ValueError('CAD part IDs disagree with manifest')
    if df[['part_id','face_id']].duplicated().any() or df.isna().any().any():
        raise ValueError('Duplicate CAD face rows or missing values')
    if not df.label.astype(int).between(0,15).all():
        raise ValueError('Invalid labels in CSV')

    # No feature, graph or label from the held-out test set is used to fit or validate.
    train = df.loc[df.part_id.isin(splits['train'])].sort_values(['part_id','face_id']).reset_index(drop=True)
    val = df.loc[df.part_id.isin(splits['validation'])].sort_values(['part_id','face_id']).reset_index(drop=True)
    features = [c for c in df.columns if c not in ('part_id','face_id','label')]
    if 'surface_type' not in features:
        raise ValueError('Missing surface_type')
    numerics = [c for c in features if c != 'surface_type']
    preprocessing = ColumnTransformer([
        ('surface', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['surface_type']),
        ('numeric', StandardScaler(), numerics),
    ])
    x_train = preprocessing.fit_transform(train[features]).astype(np.float32)
    x_val = preprocessing.transform(val[features]).astype(np.float32)
    train_graphs = load_graphs(train, x_train, args.graphs)
    val_graphs = load_graphs(val, x_val, args.graphs)
    # CONTROLLED ABLATION: remove all edges from both datasets AFTER verifying
    # that their original per-part node IDs matched the frozen label CSV.
    # The GraphSAGE architecture remains IDENTICAL; its neighbor mean is zero.
    empty_edges = torch.empty((2, 0), dtype=torch.int64)
    for g in train_graphs:
        g.edges = empty_edges
    for g in val_graphs:
        g.edges = empty_edges
    assert sum(g.edges.numel() for g in train_graphs + val_graphs) == 0
    print('ABLATION MODE: all CAD adjacency edges disabled in TRAIN and VALIDATION.', flush=True)
    if len(train_graphs) != len(splits['train']) or len(val_graphs) != len(splits['validation']):
        raise ValueError('Graph count not aligned to split')
    train_dl = DataLoader(GraphDataset(train_graphs), batch_size=args.batch_parts,
                          shuffle=True, collate_fn=collate_graphs, num_workers=0)
    val_dl = DataLoader(GraphDataset(val_graphs), batch_size=args.batch_parts,
                        shuffle=False, collate_fn=collate_graphs, num_workers=0)

    model = FaceGraphSAGE(x_train.shape[1], hidden_dim=args.hidden).to(device)
    count = np.bincount(train.label.to_numpy(dtype=np.int64),minlength=N_CLASSES).astype(np.float64)
    if (count==0).any():
        raise ValueError('Missing a class from training partition')
    w = np.sqrt(count.mean()/count)
    w = np.minimum(w,5.0)
    weights = torch.tensor(w/w.mean(), dtype=torch.float32,device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 0
    patience_left = args.patience
    best_state = None
    print(f'Device: {device}; train: {len(train_graphs)} parts / {len(train)} faces; validation: {len(val_graphs)} parts / {len(val)} faces',flush=True)
    print('Final TEST set excluded from training and evaluation.',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x,y,edges in train_dl:
            x,y,edges = x.to(device),y.to(device),edges.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x,edges),y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(y)
            seen += len(y)
        val_macro, actual, pred = eval_epoch(model,val_dl,device)
        print(f'Epoch {epoch:02d}: train loss={loss_sum/seen:.5f}, validation macro F1={val_macro:.4f}',flush=True)
        if val_macro > best_score + 1e-5:
            best_score=val_macro
            best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience_left=args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print('Early stopping.')
                break
    if best_state is None:
        raise RuntimeError('No successful training epoch')
    model.load_state_dict(best_state)
    _, actual, pred = eval_epoch(model,val_dl,device)
    metrics = {
        'algorithm': 'Pure PyTorch two-layer GraphSAGE: NO ADJACENCY CONTROL',
        'adjacency_mode': 'all_edges_removed_train_and_validation',
        'validation_accuracy':float(accuracy_score(actual,pred)),
        'validation_balanced_accuracy':float(balanced_accuracy_score(actual,pred)),
        'validation_macro_f1':float(f1_score(actual,pred,labels=range(16),average='macro',zero_division=0)),
        'validation_weighted_f1':float(f1_score(actual,pred,labels=range(16),average='weighted',zero_division=0)),
        'n_faces':len(actual),'n_parts':len(val_graphs),'training_parts':len(train_graphs),
        'training_faces':len(train),'best_epoch':best_epoch,
        'manifest_sha256':hashlib.sha256(raw_manifest).hexdigest(),
        'input_features':features,'input_dim':x_train.shape[1],
        'hidden_dim':args.hidden,'seed':args.seed,
        'note':'NO-EDGE ablation: identical GraphSAGE topology, features and training method; all neighbor edges removed. Validation-only. Test never evaluated. Filename-family grouping provisional.',
    }
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'input_dim':x_train.shape[1], 'hidden_dim':args.hidden},args.out/'graphsage_state.pt')
    joblib.dump(preprocessing,args.out/'training_only_preprocessor.joblib')
    (args.out/'validation_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (args.out/'validation_per_class.json').write_text(json.dumps(classification_report(actual,pred,labels=list(range(16)),target_names=FEATURE_CLASSES,output_dict=True,zero_division=0),indent=2),encoding='utf-8')
    pd.DataFrame(confusion_matrix(actual,pred,labels=list(range(16))),index=FEATURE_CLASSES,columns=FEATURE_CLASSES).to_csv(args.out/'validation_confusion.csv',index_label='actual_class')
    print('\n=== GRAPHSAGE NO-EDGE ABLATION VALIDATION ===')
    print(json.dumps(metrics,indent=2))
    print('Artifacts:',args.out)
    print('TEST PARTITION STILL UNTOUCHED.')


if __name__ == '__main__':
    main()
