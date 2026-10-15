import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(new URL('../build_persian_deck.mjs', import.meta.url));

test('help works without loading authoring dependencies', () => {
  const result = spawnSync(process.execPath, [script, '--runtime-dir', '/does-not-exist', '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0);
  for (const text of ['22-slide', 'two literature-review slides surveying recent methods', 'Usage:', '--output-dir PATH', '--build-dir PATH', '--basename NAME', '--runtime-dir PATH', '--skill-dir PATH', '--no-pdf', 'default:', 'never overwritten', 'Vazirmatn Regular and Bold', 'English-deck image crops without translation']) {
    assert.ok(result.stdout.includes(text), `Missing help text: ${text}`);
  }
  assert.equal(result.stderr, '');
});

test('method review adds two general, cited Persian slides and preserves prior slide content', () => {
  const check = String.raw`from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import re, sys
root=Path(sys.argv[1])/'persian'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def slide(z, n):
    return E.fromstring(z.read(f'ppt/slides/slide{n}.xml'))
def text(el):
    return [t.text or '' for t in el.findall('.//a:t', ns)]
def content(el):
    return [t for t in text(el) if not re.fullmatch(r'[۰-۹]+ از [۰-۹]+', t)]
with ZipFile(root/'robust_persian_asr_thesis_defense_fa_revised.pptx') as old, ZipFile(root/'robust_persian_asr_thesis_defense_fa_literature_methods.pptx') as new:
    assert len([n for n in new.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml', n)]) == 22
    for n in range(1, 21):
        shifted=n if n <= 2 else n+2
        actual=slide(new, shifted)
        assert content(slide(old, n)) == content(actual), f'Changed original slide {n}'
        assert f'{str(shifted).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))} از ۲۲' in text(actual)
    titles={3:'آموزش، سازگارسازی و اصلاح خروجی', 4:'بهسازی گفتار و ادغام نماها'}
    keys={3:['pesoparada2022pmct','liu2025acoustic','niu2025mcat','namdarzadeh2026sps','amirivojdan2026neyshekar','bhogale2024pseudolabel','bandarupalli2025efficient','baik2026dash','hu2024listen','udagawa2024conservative','shah2025srb','saijo2025urgent','sedghiyeh2025psrb'],
          4:['nasretdinov2025schrodinger','yang2026decoupling','dissen2025frontend','sato2025generic','yang2023fathubert','hu2024wav2code','wang2024bridge','cui2025realbridge','li2026oa','qu2026rethink']}
    for n in [3,4]:
        review=slide(new, n)
        assert titles[n] in text(review)
        assert len(review.findall('.//a:tbl', ns)) == 1
        assert len(review.findall('.//a:tr', ns)) == 5
        assert len(review.findall('.//a:tc', ns)) == 15
        for value in text(review):
            assert 'پرسش این پایان‌نامه' not in value
            assert 'تمرکز پژوهش:' not in value
            assert 'این پایان‌نامه' not in value
        for p in review.findall('.//a:p', ns):
            value=''.join(text(p))
            if value.strip():
                expected='1' if re.search('[\u0600-\u06ff]', value) else '0'
                assert p.find('a:pPr', ns).get('rtl') == expected, value
        notes='\n'.join(text(E.fromstring(new.read(f'ppt/notesSlides/notesSlide{n}.xml'))))
        for key in keys[n]+['Thesis/chapters/literature.tex']:
            assert key in notes, key
    assert 'Acoustic-aware Aug. (2025*)' in '\n'.join(text(slide(new,3)))
    assert 'Neyshekar (2026*)' in '\n'.join(text(slide(new,3)))
    assert 'PSRB (2025*)' in '\n'.join(text(slide(new,3)))
    assert 'Conf-OA / Rethink-and-Refine (2026)' in '\n'.join(text(slide(new,4)))
`;
  const result = spawnSync('python3', ['-c', check, fileURLToPath(new URL('..', import.meta.url))], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
});

test('implementation crops are exact copies of the English deck images', () => {
  const check = String.raw`from pathlib import Path
from zipfile import ZipFile
import json, hashlib, sys
root=Path(sys.argv[1]); assets=root/'assets/english-implementation-crops'
manifest=json.loads((assets/'provenance.json').read_text())
with ZipFile(root/'robust_persian_asr_thesis_defense_updated.pptx') as source:
    assert len(manifest['assets']) == 9
    for item in manifest['assets']:
        local=(assets/item['file']).read_bytes()
        assert local == source.read(item['source_member']), item['file']
        assert hashlib.sha256(local).hexdigest() == item['sha256'], item['file']
`;
  const result = spawnSync('python3', ['-c', check, fileURLToPath(new URL('..', import.meta.url))], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
});

test('invalid options fail before authoring', () => {
  for (const args of [['--unknown'], ['--output-dir'], ['--basename', '../overwrite-source']]) {
    const result = spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });
    assert.equal(result.status, 2);
    assert.ok(result.stderr.length > 0);
  }
});
