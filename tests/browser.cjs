const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
(async()=>{
 const browser=await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_EXECUTABLE || undefined, args:["--disable-dev-shm-usage","--disable-gpu"]});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});
 const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8765');
 await page.getByRole('button',{name:'Try a small example'}).click();
 await page.waitForSelector('.pair');
 assert.equal(await page.locator('.pair').count(),10);
 await page.locator('#decision-0').selectOption('duplicate');
 const downloadPromise=page.waitForEvent('download');
 await page.getByRole('button',{name:'Export review'}).click();
 const download=await downloadPromise;
 const json=JSON.parse(await fs.readFile(await download.path(),'utf8'));
 assert.equal(json.reviews[0].decision,'duplicate');
 assert.equal(json.method,'tfidf');
 await page.screenshot({path:'reports/desktop.png',fullPage:false});
 // Real upload path, both inference methods, truncation warnings.
 await page.locator('#file').setInputFiles('examples/issues.csv');
 await page.locator('#method').selectOption('minilm');
 await page.locator('#top_k').fill('10');
 await page.getByRole('button',{name:'Find candidate pairs'}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('MiniLM semantic'));
 assert.equal(await page.locator('.pair').count(),10);
 assert.equal(await page.locator('#decision-0').inputValue(),'unreviewed');
 // Render imported markup as text and ensure errors clear previous export.
 await page.locator('#file').setInputFiles({name:'xss.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify([
  {id:1,title:'<img src=x onerror=alert(1)>',body:'</p><script>alert(1)</script>'},
  {id:2,title:'Other issue',body:'A different problem'}
 ]))});
 await page.locator('#method').selectOption('tfidf');
 await page.locator('#min_score').fill('-1');
 await page.getByRole('button',{name:'Find candidate pairs'}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('1 pairs shown'));
 assert.equal(await page.locator('.pair img,.pair script').count(),0);
 await page.locator('#file').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from('{broken')});
 await page.getByRole('button',{name:'Find candidate pairs'}).click();
 await page.waitForSelector('#status.error');
 assert.equal(await page.locator('#download').isDisabled(),true);
 assert.equal(await page.locator('.pair').count(),0);
 await page.setViewportSize({width:390,height:844});
 await page.getByRole('button',{name:'Try a small example'}).click();
 await page.waitForSelector('.pair');
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
 await page.screenshot({path:'reports/mobile.png',fullPage:false});
 assert.deepEqual(errors,[]);
 console.log('Browser checks passed: demo, JSON review export, CSV MiniLM upload, reset, XSS-as-text, malformed input, mobile layout.');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
