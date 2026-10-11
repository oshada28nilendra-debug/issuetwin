import json
import numpy as np
import pytest
from issuetwin.core import parse_issues, rank_pairs, tfidf_embeddings, evaluate_pairs


def issues(rows=None):
    return parse_issues(json.dumps(rows or [
        {'number': 1, 'title': 'Login fails', 'body': 'Password reset crashes'},
        {'number': 2, 'title': 'Login fails', 'body': 'Password reset crashes'},
        {'number': 3, 'title': 'Export chart', 'body': 'Download a plot'},
    ]).encode(), 'issues.json')


def test_github_export_skips_pull_requests_and_normalizes_null_body():
    result = issues([{'number': 8, 'title': ' Bug ', 'body': None},
                     {'number': 9, 'title': 'PR', 'pull_request': {}}])
    assert len(result) == 1 and result[0].id == '8'
    assert result[0].title == 'Bug' and result[0].body == ''


def test_csv_quoted_multiline_bom_and_gh_list_shape():
    csv = '\ufeffid,title,body\r\nA,"Comma, title","line1\nline2"\r\n'
    got = parse_issues(csv.encode(), 'export.CSV')
    assert got[0].body == 'line1\nline2'
    assert issues([{'number': 2, 'title': 'hello', 'url': 'https://github.com/a/b/issues/2'}])[0].url.endswith('/2')


@pytest.mark.parametrize('rows,match', [
    ([{'id': 'a', 'title': ''}], 'title'),
    ([{'id': 'a', 'title': 'ok'}, {'id': 'a', 'title': 'other'}], 'Duplicate'),
    ([{'title': 'no id'}], 'id'),
    ([{'id': True, 'title': 'bad'}], 'id'),
    ([{'id': 'a', 'title': 'ok', 'body': ['not text']}], 'body'),
    ([{'id': 'a', 'title': 'ok', 'url': 'javascript:alert(1)'}], 'URL'),
    ([], 'issue'),
])
def test_invalid_records_are_rejected(rows, match):
    with pytest.raises(ValueError, match=match):
        parse_issues(json.dumps(rows).encode(), 'a.json')


@pytest.mark.parametrize('data,name', [(b'{broken', 'a.json'), (b'{}', 'a.json'),
    (b'\xff', 'a.csv'), (b'id,title\nx,ok,extra', 'a.csv'),
    (b'id,id,title\nx,x,ok', 'a.csv'), (b'[]', 'a.exe')])
def test_invalid_files(data, name):
    with pytest.raises(ValueError):
        parse_issues(data, name)


def test_limits():
    with pytest.raises(ValueError): parse_issues(b' ' * (5 * 1024 * 1024 + 1), 'a.json')
    with pytest.raises(ValueError): issues([{'id': str(i), 'title': 'ok'} for i in range(501)])


def test_rank_normalizes_vectors_omits_self_and_reverse_pairs():
    ranked = rank_pairs(issues(), np.array([[2,0], [3,0], [0,4]]))
    assert len(ranked) == 3
    assert (ranked[0]['left_id'], ranked[0]['right_id']) == ('1', '2')
    assert ranked[0]['score'] == pytest.approx(1)
    assert len(rank_pairs(issues(), np.eye(3), top_k=1)) == 1
    assert rank_pairs(issues(), np.eye(3), min_score=.1) == []


def test_ties_are_stable_by_identifier_even_after_reordering():
    a=issues(); b=list(reversed(a))
    assert rank_pairs(a, np.ones((3,2))) == rank_pairs(b, np.ones((3,2)))


@pytest.mark.parametrize('vectors', [np.ones((2,3)), np.array([[1,0],[0,0],[0,1]]), np.full((3,2),np.nan)])
def test_invalid_vectors_fail(vectors):
    with pytest.raises(ValueError): rank_pairs(issues(),vectors)


def test_tfidf_handles_repeated_and_punctuation_only_documents():
    ranked = rank_pairs(issues(),tfidf_embeddings(issues()))
    assert ranked[0]['score'] == pytest.approx(1)
    with pytest.raises(ValueError, match='vocabulary'):
        tfidf_embeddings(issues([{'id': 'x', 'title': '!!!'}, {'id':'y','title':'???'}]))


def test_hand_calculated_metrics_and_exhaustive_label_requirement():
    pairs=[{'left_id':'a','right_id':'b','score':.9},
           {'left_id':'a','right_id':'c','score':.8},
           {'left_id':'b','right_id':'c','score':.2}]
    labels=[{'left_id':'a','right_id':'b','duplicate':True},
            {'left_id':'a','right_id':'c','duplicate':False},
            {'left_id':'b','right_id':'c','duplicate':True}]
    m=evaluate_pairs(pairs,labels,k=2,threshold=.7)
    assert m['average_precision'] == pytest.approx((1+2/3)/2)
    assert m['precision_at_k'] == .5 and m['recall_at_k'] == .5
    assert m['confusion'] == {'tp':1,'fp':1,'fn':1,'tn':0}
    with pytest.raises(ValueError): evaluate_pairs(pairs,labels[:-1])
    with pytest.raises(ValueError): evaluate_pairs(pairs,labels+labels[:1])
    with pytest.raises(ValueError): evaluate_pairs(pairs,[dict(x,duplicate='false') for x in labels])


def test_blank_pull_request_markers_keep_normal_issues():
    csv = b'id,title,pull_request\n1,Normal issue,\n2,A pull request,true\n'
    assert [x.id for x in parse_issues(csv, 'a.csv')] == ['1']
    data = json.dumps([{'number': 1, 'title': 'Normal', 'pull_request': None},
                       {'number': 2, 'title': 'PR', 'pull_request': {}}]).encode()
    assert [x.id for x in parse_issues(data, 'a.json')] == ['1']
