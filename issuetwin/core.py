"""Validated input, deterministic ranking and exhaustive pair evaluation."""
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

MAX_BYTES = 5 * 1024 * 1024
MAX_ISSUES = 500
MAX_TEXT = 20000


@dataclass(frozen=True)
class Issue:
    id: str
    title: str
    body: str = ''
    url: str = ''

    @property
    def text(self):
        return self.title + '\n' + self.body


def parse_issues(data: bytes, filename: str) -> list[Issue]:
    if len(data) > MAX_BYTES:
        raise ValueError('Export exceeds the 5 MiB limit.')
    try:
        text = data.decode('utf-8-sig')
        suffix = Path(filename).suffix.lower()
        if suffix == '.json':
            rows = json.loads(text)
            if not isinstance(rows, list):
                raise ValueError('JSON must be an array of issue objects.')
        elif suffix == '.csv':
            reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)):
                raise ValueError('CSV has duplicate column names.')
            rows = list(reader)
            if any(None in row or any(v is None for v in row.values()) for row in rows):
                raise ValueError('CSV row width does not match its header.')
        else:
            raise ValueError('Use a .json or .csv export.')
    except (UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError('Invalid UTF-8 JSON/CSV export: ' + str(exc)) from exc
    if len(rows) > MAX_ISSUES:
        raise ValueError(f'Export contains more than {MAX_ISSUES} records.')
    result, seen = [], set()
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Row {i} must be an issue object.')
        pr = row.get('pull_request')
        is_pr = (pr is not None) if suffix == '.json' else bool(str(pr or '').strip())
        if is_pr:
            continue
        ident = row.get('number', row.get('id'))
        if isinstance(ident, bool) or not isinstance(ident, (str, int)) or not str(ident).strip():
            raise ValueError(f'Row {i} needs a non-empty id or number.')
        ident = str(ident).strip()
        if len(ident) > 100:
            raise ValueError(f'Row {i} id exceeds 100 characters.')
        if ident in seen:
            raise ValueError(f'Duplicate issue id: {ident}. Import one repository at a time.')
        title, body = row.get('title'), row.get('body')
        body = '' if body is None else body
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f'Row {i} needs a non-empty title.')
        if not isinstance(body, str):
            raise ValueError(f'Row {i} body must be text or null.')
        if len(title) + len(body) > MAX_TEXT:
            raise ValueError(f'Row {i} title/body exceeds {MAX_TEXT} characters.')
        url = row.get('html_url') or row.get('url') or ''
        if not isinstance(url, str):
            raise ValueError(f'Row {i} URL must be text.')
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ('https', 'http') or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError(f'Row {i} URL must be an absolute HTTP(S) URL without credentials.')
        result.append(Issue(ident, title.strip(), body.strip(), url))
        seen.add(ident)
    if not result:
        raise ValueError('No issues found (pull requests are excluded).')
    return result


def tfidf_embeddings(issues):
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                max_features=20000, dtype=np.float32).fit_transform([x.text for x in issues])
    except ValueError as exc:
        raise ValueError('No TF-IDF vocabulary: use words with at least two characters.') from exc
    if np.any(np.asarray(matrix.getnnz(axis=1)) == 0):
        raise ValueError('An issue has no TF-IDF vocabulary. Use words with at least two characters.')
    return matrix.toarray()


def rank_pairs(issues, embeddings, top_k=None, min_score=-1):
    if not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not -1 <= min_score <= 1:
        raise ValueError('Minimum score must be finite and between -1 and 1.')
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError('top_k must be a positive integer.')
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(issues) or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError('Embedding matrix has invalid shape or non-finite values.')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Embedding matrix contains a zero vector.')
    x = x / norms
    scores = np.clip(x @ x.T, -1, 1)
    pairs = []
    for i in range(len(issues)):
        for j in range(i + 1, len(issues)):
            score = float(scores[i, j])
            if score >= min_score:
                left, right = sorted((issues[i].id, issues[j].id))
                pairs.append({'left_id': left, 'right_id': right, 'score': score})
    pairs.sort(key=lambda p: (-p['score'], p['left_id'], p['right_id']))
    return pairs if top_k is None else pairs[:top_k]


def evaluate_pairs(pairs, labels, k=10, threshold=.7):
    if type(k) is not int or k < 1 or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Use positive k and a finite threshold in [-1, 1].')
    if not pairs or not isinstance(labels, list):
        raise ValueError('Evaluation requires ranked pairs and exhaustive labels.')
    truth = {}
    for label in labels:
        if not isinstance(label, dict) or type(label.get('duplicate')) is not bool:
            raise ValueError('Each label must contain a boolean duplicate value.')
        a, b = str(label.get('left_id', '')), str(label.get('right_id', ''))
        key = tuple(sorted((a, b)))
        if not a or not b or a == b or key in truth:
            raise ValueError('Invalid or repeated labelled pair.')
        truth[key] = label['duplicate']
    keys = [tuple(sorted((p['left_id'], p['right_id']))) for p in pairs]
    if len(set(keys)) != len(keys) or set(keys) != set(truth):
        raise ValueError('Labels must cover every candidate pair exactly once, with no extra pairs.')
    ranked = sorted(pairs, key=lambda p: (-p['score'], p['left_id'], p['right_id']))
    y = np.array([truth[tuple(sorted((p['left_id'], p['right_id'])))] for p in ranked], dtype=int)
    scores = np.array([p['score'] for p in ranked])
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite evaluation score.')
    positives = int(y.sum())
    n = min(k, len(ranked))
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = positives - tp
    tn = len(y) - positives - fp
    return {'pair_count': len(y), 'positive_pairs': positives,
            'average_precision': float(average_precision_score(y, scores)) if positives else None,
            'k_requested': k, 'k_effective': n,
            'precision_at_k': float(y[:n].sum() / n),
            'recall_at_k': float(y[:n].sum() / positives) if positives else None,
            'threshold': threshold, 'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'threshold_precision': tp / (tp + fp) if tp + fp else None,
            'threshold_recall': tp / positives if positives else None}
