import hashlib
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone

from .core import parse_issues, rank_pairs, tfidf_embeddings


def analyze(data, filename, method='minilm', top_k=30, min_score=-1):
    if method not in ('minilm', 'tfidf'):
        raise ValueError('Choose minilm or tfidf.')
    if type(top_k) is not int or not 1 <= top_k <= 1000:
        raise ValueError('Show between 1 and 1000 pairs.')
    if not math.isfinite(min_score) or not -1 <= min_score <= 1:
        raise ValueError('Minimum score must be finite and in [-1, 1].')
    issues = parse_issues(data, filename)
    if len(issues) < 2:
        raise ValueError('Import at least two issues to find candidate pairs.')
    warnings = ['Scores are cosine similarities, not duplicate probabilities. Review every suggested pair.']
    started = time.perf_counter()
    if method == 'minilm':
        from .model import encode_issues
        vectors, metadata = encode_issues(issues)
        if metadata['truncated_issue_ids']:
            warnings.append(f"MiniLM truncated {len(metadata['truncated_issue_ids'])} issue(s) at 256 tokens; see model metadata for IDs.")
    else:
        vectors = tfidf_embeddings(issues)
        metadata = {'model_id': 'corpus-fitted TF-IDF', 'ngram_range': [1, 2],
                    'sublinear_tf': True, 'max_features': 20000, 'truncated_issue_ids': []}
    pairs = rank_pairs(issues, vectors, top_k, min_score)
    return {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'method': method, 'model': metadata, 'input_sha256': hashlib.sha256(data).hexdigest(),
            'issue_count': len(issues), 'candidate_count': len(issues) * (len(issues)-1)//2,
            'top_k': top_k, 'min_score': min_score, 'seconds': time.perf_counter()-started,
            'warnings': warnings, 'issues': [asdict(x) for x in issues], 'pairs': pairs}
