import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(new URL('../build_persian_deck.mjs', import.meta.url));

test('help works without loading authoring dependencies', () => {
  const result = spawnSync(process.execPath, [script, '--runtime-dir', '/does-not-exist', '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0);
  for (const text of ['Usage:', '--output-dir PATH', '--build-dir PATH', '--basename NAME', '--runtime-dir PATH', '--skill-dir PATH', '--no-pdf', 'default:', 'never overwritten', 'Vazirmatn Regular and Bold', 'English-deck image crops without translation']) {
    assert.ok(result.stdout.includes(text), `Missing help text: ${text}`);
  }
  assert.equal(result.stderr, '');
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
