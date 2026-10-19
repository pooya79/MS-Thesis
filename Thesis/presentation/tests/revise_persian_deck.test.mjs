import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const script=fileURLToPath(new URL('../revise_persian_deck.mjs',import.meta.url));
test('review CLI documents arguments and preserves outputs',()=>{
  const r=spawnSync(process.execPath,[script,'--help'],{encoding:'utf8'});
  assert.equal(r.status,0,r.stderr);
  for(const key of ['40-slide','--source PATH','--output-dir PATH','--basename NAME','--build-dir PATH','--pdf-renderer PATH','--no-pdf','default:','never overwritten']) assert.ok(r.stdout.includes(key),key);
});
test('review CLI rejects unknown and missing arguments',()=>{
  for(const args of [['--unknown'],['--source'],['--basename','../bad']]) {
    const r=spawnSync(process.execPath,[script,...args],{encoding:'utf8'});assert.equal(r.status,2);
  }
});
test('reviewed deck preserves evidence and includes units and native research bullets',()=>{
  const check=String.raw`
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as E
import re,sys
root=Path(sys.argv[1]); ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with ZipFile(root/'robust_persian_asr_thesis_defense_fa_references_qwen_compact.pptx') as old, ZipFile(root/'robust_persian_asr_thesis_defense_fa_reviewed.pptx') as new:
    def slide(n): return E.fromstring(new.read(f'ppt/slides/slide{n}.xml'))
    def text(n): return ''.join(slide(n).itertext())
    p=E.fromstring(new.read('ppt/presentation.xml'))
    ids=p.find('p:sldIdLst',ns);assert len(ids)==40
    assert ids[23].get('{'+ns['r']+'}id')=='rIdPersianReview'
    for name in old.namelist():
        if name.startswith(('ppt/embeddings/','ppt/media/')): assert new.read(name)==old.read(name),name
        if name.startswith('ppt/charts/'):
            if not name.endswith('.xml'): assert new.read(name)==old.read(name),name
            else:
                def evidence(content):
                    root=E.fromstring(content)
                    chart='http://schemas.openxmlformats.org/drawingml/2006/chart'
                    return ([v.text for v in root.findall('.//{'+chart+'}v')],
                            [v.text for v in root.findall('.//a:t',ns)])
                assert evidence(new.read(name))==evidence(old.read(name)),name
    for n in [6,37]:
        assert 'آموزش (نمونه)' in text(n) and 'اعتبارسنجی (نمونه)' in text(n)
        assert 'Utterance' in text(n)
    assert 'WER (%) = 100 × (S + D + I) / N' in text(7)
    assert text(7).count('نمونه')>=6
    assert 'واحد درصد' in text(20) and 'بر حسب درصد' not in text(20)
    assert '12.2 kb/s' in text(32) and '12.65 kb/s' in text(32)
    assert 'بی‌واحد' in text(36) and 'unitless' in text(39)
    assert '۰٫۰۰۰۱' in text(13) and '۰٫۰۰۰۰۰۵' in text(13)
    assert not re.search('[\u2066-\u2069?؟\ufffd]',text(13))
    for n in [23,40]:
        assert len(slide(n).findall('.//a:buChar',ns))==4
        for p in slide(n).findall('.//a:p',ns):
            if p.find('a:pPr/a:buChar',ns) is not None: assert p.find('a:pPr',ns).get('rtl')=='1'
    assert 'نتیجه‌گیری پژوهش' in text(23) and 'نوآوری‌ها و دستاوردهای پژوهش' in text(40)
    assert '۹٫۸۳ واحد درصد' in text(23) and '۰٫۴۳ واحد درصد' in text(23)
    for n in [24,25,26]:
        original=E.fromstring(old.read(f'ppt/slides/slide{n}.xml'))
        assert list(slide(n).itertext())==list(original.itertext())
    for n in [27,28,29,30,31,33,34,35,38]:
        assert new.read(f'ppt/slides/slide{n}.xml')==old.read(f'ppt/slides/slide{n}.xml'),n
`;
  const r=spawnSync('python3',['-c',check,fileURLToPath(new URL('../persian',import.meta.url))],{encoding:'utf8'});assert.equal(r.status,0,r.stderr);
});
