const form = document.querySelector('#analyze-form');
const status = document.querySelector('#status');
const pairsRoot = document.querySelector('#pairs');
let report = null;
const decisions = new Map();
const el = (tag, text, cls) => { const node = document.createElement(tag); if (text != null) node.textContent = text; if (cls) node.className = cls; return node; };
function render() {
  pairsRoot.replaceChildren();
  const lookup = new Map(report.issues.map(x => [x.id, x]));
  for (const [index, pair] of report.pairs.entries()) {
    const card = el('article', null, 'pair');
    const head = el('div', null, 'pair-head');
    head.append(el('span', `PAIR ${String(index + 1).padStart(2, '0')}`), el('span', `Cosine ${pair.score.toFixed(3)}`, 'score'));
    const comparison = el('div', null, 'comparison');
    for (const id of [pair.left_id, pair.right_id]) {
      const issue = lookup.get(id), block = el('div', null, 'issue');
      block.append(el('small', `ISSUE ${id}`), el('h3', issue.title), el('p', issue.body || 'No description supplied.', 'body'));
      if (issue.url) { const link = el('a', 'Open original ↗'); link.href = issue.url; link.target = '_blank'; link.rel = 'noopener noreferrer'; block.append(link); }
      comparison.append(block);
    }
    const review = el('div', null, 'review');
    const label = el('label', 'Your decision'); label.htmlFor = `decision-${index}`;
    const select = el('select'); select.id = label.htmlFor;
    for (const [value, text] of [['unreviewed','Not reviewed'],['duplicate','Likely duplicate'],['different','Different issues'],['uncertain','Needs investigation']]) {
      const option = el('option', text); option.value = value; select.append(option);
    }
    select.addEventListener('change', () => decisions.set(index, select.value));
    review.append(label, select); card.append(head, comparison, review); pairsRoot.append(card);
  }
  if (!report.pairs.length) pairsRoot.append(el('div', 'No pairs match this filter. Try a lower minimum score.', 'empty'));
}
async function analyze(data) {
  document.querySelector('#run').disabled = true; document.querySelector('#demo').disabled = true;
  document.querySelector('#download').disabled = true;
  report = null; decisions.clear(); pairsRoot.replaceChildren();
  document.querySelector('#stats').hidden = true; document.querySelector('#warnings').replaceChildren();
  status.className = ''; status.textContent = 'Analysing your queue… MiniLM may download its model on first use.';
  try {
    const response = await fetch('/api/analyze', {method:'POST', headers:{'X-IssueTwin':'1'}, body:data});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Analysis failed.');
    report = result;
    status.textContent = `${report.pairs.length} pairs shown · ${report.method === 'minilm' ? 'MiniLM semantic embeddings' : 'TF-IDF word overlap'}`;
    const stats = document.querySelector('#stats'); stats.hidden = false; stats.replaceChildren();
    for (const text of [`${report.issue_count} issues`, `${report.candidate_count} possible pairs`, `${report.seconds.toFixed(2)}s analysis`]) stats.append(el('span', text));
    const warnings = document.querySelector('#warnings');
    for (const text of report.warnings) warnings.append(el('p', text, 'warning'));
    render(); document.querySelector('#download').disabled = false;
  } catch (error) { status.className = 'error'; status.textContent = error.message; }
  finally { document.querySelector('#run').disabled = false; document.querySelector('#demo').disabled = false; }
}
form.addEventListener('submit', event => { event.preventDefault(); analyze(new FormData(form)); });
document.querySelector('#demo').addEventListener('click', () => {
  const data = new FormData(form);
  data.set('file', new Blob([JSON.stringify([
    {number:1,title:'Password reset link expires immediately',body:'The recovery email arrives, but clicking the link says it is expired.'},
    {number:2,title:'Recovery email contains an expired reset link',body:'Cannot reset my password because the link is already expired when received.'},
    {number:3,title:'CSV export leaves out the final row',body:'Exporting 100 rows produces a file containing only 99 records.'},
    {number:4,title:'Last record is missing from CSV download',body:'The CSV export drops the final row in the table.'},
    {number:5,title:'Add dark mode to settings',body:'Please provide a dark colour theme.'}
  ])], {type:'application/json'}), 'synthetic-example.json');
  data.set('method','tfidf'); document.querySelector('#method').value = 'tfidf'; analyze(data);
});
document.querySelector('#download').addEventListener('click', () => {
  if (!report) return;
  const output = {...report, reviews:report.pairs.map((p,i) => ({left_id:p.left_id,right_id:p.right_id,decision:decisions.get(i)||'unreviewed'}))};
  const url = URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));
  const a = el('a'); a.href=url; a.download='issuetwin-review.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
