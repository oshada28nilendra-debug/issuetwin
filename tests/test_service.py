import io
import json
import pytest
from issuetwin.service import analyze
from issuetwin.web import create_app

DATA = json.dumps([{'id':'a','title':'Login fails'},{'id':'b','title':'Login fails'},
                   {'id':'c','title':'Export a chart'}]).encode()


def test_report_includes_provenance_and_pair_limit():
    report=analyze(DATA,'a.json','tfidf',1,0)
    assert len(report['pairs'])==1 and report['candidate_count']==3
    assert report['method']=='tfidf' and len(report['input_sha256'])==64
    assert len(report['issues'])==3 and report['warnings']
    assert report['pairs'][0]['score']==pytest.approx(1)


@pytest.mark.parametrize('method,k,score',[('fake',5,0),('tfidf',0,0),('tfidf',5,float('nan'))])
def test_service_rejects_bad_options(method,k,score):
    with pytest.raises(ValueError): analyze(DATA,'a.json',method,k,score)


def test_server_round_trip_and_validation():
    client=create_app().test_client()
    assert client.get('/').status_code==200
    assert client.post('/api/analyze').status_code==403
    response=client.post('/api/analyze',headers={'X-IssueTwin':'1'},data={
        'file':(io.BytesIO(DATA),'issues.json'),'method':'tfidf','top_k':'10','min_score':'0'})
    assert response.status_code==200 and len(response.json['pairs'])==3
    assert response.headers['Cache-Control']=='no-store'
    assert client.post('/api/analyze',headers={'X-IssueTwin':'1'},data={}).status_code==400
    blocked=client.post('/api/analyze',headers={'X-IssueTwin':'1','Origin':'https://evil.example'})
    assert blocked.status_code==403
    assert client.get('/',headers={'Host':'evil.example'}).status_code==400


@pytest.mark.model
def test_real_model_recognizes_paraphrase_and_reports_truncation():
    from issuetwin.model import encode_issues
    from issuetwin.core import parse_issues,rank_pairs
    data=json.dumps([{'id':'a','title':'Password reset link has expired'},
        {'id':'b','title':'Recovery email URL is no longer valid'},
        {'id':'c','title':'Change the chart background to green'},
        {'id':'d','title':'Long report','body':'hello '*400}]).encode()
    issues=parse_issues(data,'a.json')
    vectors,meta=encode_issues(issues)
    assert vectors.shape==(4,384)
    assert meta['truncated_issue_ids']==['d']
    pairs=rank_pairs(issues[:3],vectors[:3])
    assert (pairs[0]['left_id'],pairs[0]['right_id'])==('a','b')
