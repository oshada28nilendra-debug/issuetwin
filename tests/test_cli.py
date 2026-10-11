import json
import subprocess
import sys


def test_cli_rank_and_evaluation(tmp_path):
    issues=tmp_path/'issues.json'
    issues.write_text(json.dumps([{'id':'a','title':'Login fails'}, {'id':'b','title':'Login fails'}, {'id':'c','title':'Export data'}]))
    labels=tmp_path/'labels.json'
    labels.write_text(json.dumps([{'left_id':'a','right_id':'b','duplicate':True},
        {'left_id':'a','right_id':'c','duplicate':False},{'left_id':'b','right_id':'c','duplicate':False}]))
    report=tmp_path/'report.json'
    command=[sys.executable,'-m','issuetwin.cli']
    run=subprocess.run(command+['rank',str(issues),'--method','tfidf','--output',str(report)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['pairs'][0]['score']>.99
    run=subprocess.run(command+['evaluate',str(issues),str(labels),'--methods','tfidf','--output',str(report)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['methods']['tfidf']['metrics']['average_precision']==1
    run=subprocess.run(command+['rank','nonexistent.json'],capture_output=True,text=True)
    assert run.returncode==2 and 'Error:' in run.stderr
