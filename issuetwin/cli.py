"""Commands for local review, ranking and reproducible pair evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import MAX_BYTES, evaluate_pairs, parse_issues, rank_pairs, tfidf_embeddings
from .service import analyze


def read_bounded(path):
    with Path(path).open('rb') as handle:
        content = handle.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError('Input exceeds 5 MiB.')
    return content


def write_json(value, output):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding='utf-8')
        print(f'Wrote {output}')
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description='IssueTwin: review candidate duplicate issues locally.')
    commands = parser.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve', help='Open the local browser interface')
    serve.add_argument('--port', type=int, default=8765)
    commands.add_parser('download', help='Download and cache the pinned MiniLM assets')
    rank = commands.add_parser('rank', help='Rank pairs from a JSON or CSV export')
    rank.add_argument('issues')
    rank.add_argument('--method', choices=['minilm', 'tfidf'], default='minilm')
    rank.add_argument('--top-k', type=int, default=30)
    rank.add_argument('--min-score', type=float, default=-1)
    rank.add_argument('--output')
    evaluate = commands.add_parser('evaluate', help='Compare methods against exhaustive binary pair labels')
    evaluate.add_argument('issues')
    evaluate.add_argument('labels')
    evaluate.add_argument('--methods', nargs='+', choices=['minilm', 'tfidf'], default=['minilm', 'tfidf'])
    evaluate.add_argument('--k', type=int, default=10)
    evaluate.add_argument('--threshold', type=float, default=.7)
    evaluate.add_argument('--dataset-note', default='User-supplied data; provenance not independently verified.')
    evaluate.add_argument('--output')
    args = parser.parse_args()
    try:
        if args.command == 'serve':
            if not 1 <= args.port <= 65535:
                raise ValueError('Port must be in 1..65535.')
            from .web import create_app
            create_app().run(host='127.0.0.1', port=args.port, debug=False)
        elif args.command == 'download':
            from .model import _load, MODEL_ID, MODEL_REVISION
            _load()
            print(f'Cached {MODEL_ID}@{MODEL_REVISION}')
        elif args.command == 'rank':
            write_json(analyze(read_bounded(args.issues), args.issues, args.method, args.top_k, args.min_score), args.output)
        else:
            data, label_data = read_bounded(args.issues), read_bounded(args.labels)
            issues = parse_issues(data, args.issues)
            labels = json.loads(label_data)
            results = {}
            for method in dict.fromkeys(args.methods):
                start = time.perf_counter()
                if method == 'minilm':
                    from .model import encode_issues
                    vectors, metadata = encode_issues(issues)
                else:
                    vectors = tfidf_embeddings(issues)
                    metadata = {'model_id': 'corpus-fitted TF-IDF', 'ngram_range': [1, 2],
                                'max_features': 20000, 'sublinear_tf': True}
                pairs = rank_pairs(issues, vectors)
                elapsed = time.perf_counter() - start
                results[method] = {'model': metadata, 'seconds_including_model_load': elapsed,
                    'metrics': evaluate_pairs(pairs, labels, args.k, args.threshold),
                    'top_pairs': pairs[:args.k]}
            write_json({'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'dataset_note': args.dataset_note, 'issue_count': len(issues),
                'input_sha256': hashlib.sha256(data).hexdigest(),
                'labels_sha256': hashlib.sha256(label_data).hexdigest(),
                'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                    'packages': {name: importlib.metadata.version(name) for name in
                                 ['numpy','scikit-learn','onnxruntime','tokenizers','huggingface-hub']}},
                'protocol': 'All unordered pairs labelled; no label-based fitting or threshold tuning. AP groups equal-score ties; top-k ties use lexical IDs. Timings include model loading and are not steady-state benchmarks.',
                'methods': results}, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
