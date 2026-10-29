import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const dir = fileURLToPath(new URL('../', import.meta.url));
test('export help is available without browser or rendering dependencies', () => {
  const result = spawnSync(process.execPath, [`${dir}export_persian_figures.mjs`, '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  for (const option of ['--source-dir', '--output-dir', '--preview-dir', '--browser-path', '--module-dir', 'default:']) {
    assert.ok(result.stdout.includes(option), JSON.stringify({ option, stdout: result.stdout, stderr: result.stderr }));
  }
});
test('six offline Persian diagrams are linked from the thesis', () => {
  const sources = readdirSync(dir).filter(name => name.endsWith('-fa.html'));
  assert.equal(sources.length, 6);
  const thesis = readFileSync(new URL('../../chapters/work.tex', import.meta.url), 'utf8');
  for (const name of sources) {
    const html = readFileSync(dir + name, 'utf8');
    assert.match(html, /lang="fa"/);
    assert.match(html, /viewBox="0 0 1120 792"/);
    assert.match(html, /role="img" aria-labelledby=/);
    assert.match(html, /data:font\/ttf;base64,/);
    // Ignore encoded font data when checking prose and external dependencies.
    const markup = html.replace(/base64,[A-Za-z0-9+/=]+/g, 'base64,FONT');
    assert.doesNotMatch(markup, /[ءأإؤئٕٔۀ]/u);
    assert.doesNotMatch(markup, /<(?:script|link)\b/);
    assert.ok(thesis.includes(`figs/${name.replace('.html', '.pdf')}`));
  }
  const fusion = thesis.split('\\قسمت{مدل دوجریانه بهسازی-همجوشی}')[1].split('\\قسمت{روش مقایسه‌ای: پل تطبیقی}')[0];
  assert.doesNotMatch(fusion, /[ءأإؤئٕٔۀ]/u);
});

test('degradation figure retains the English branch hierarchy with readable Persian labels', () => {
  const html = readFileSync(dir + 'degradation-pipeline-architecture-fa.html', 'utf8');
  for (const node of ['source', 'preparation', 'channel', 'opus', 'ffmpeg', 'passthrough', 'alignment', 'output']) {
    assert.ok(html.includes(`data-node="${node}"`), `Missing stage: ${node}`);
  }
  for (const label of ['مسیر کدک و شبکه', 'مرجع پاک', 'هم‌ترازسازی جفت صوتی', 'Opus', 'ffmpeg', 'WAV + JSONL']) {
    assert.ok(html.includes(label), `Missing label: ${label}`);
  }
  // Both fallback branches must show that waveform dropout is conditional.
  assert.equal((html.match(/اگر اختلال شبکه فعال باشد:/g) ?? []).length, 2);
  assert.equal((html.match(/حذف قاب از موج صوتی/g) ?? []).length, 2);
  assert.match(html, /<path[^>]*stroke-dasharray="4 4"[^>]*marker-end=/);
  assert.match(html, /\.latin\s*\{[^}]*direction:\s*ltr/);
  assert.doesNotMatch(html, /<text\b[^>]*transform=/);
});

test('IranSeda figure preserves three phases and all seven numbered stages', () => {
  const html = readFileSync(dir + 'long-audio-data-creation-pipeline-fa.html', 'utf8');
  for (const phase of ['آماده‌سازی منابع', 'ساخت شبه‌برچسب', 'آموزش مدل']) {
    assert.ok(html.includes(phase), `Missing phase: ${phase}`);
  }
  const stages = [...html.matchAll(/data-node="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(stages, ['source', 'segmentation', 'transcription', 'normalization', 'refinement', 'filtering', 'training']);
  const numbers = [...html.matchAll(/class="step-number"[^>]*>([^<]+)</g)].map(match => match[1]);
  assert.deepEqual(numbers, ['۰۱', '۰۲', '۰۳', '۰۴', '۰۵', '۰۶', '۰۷']);
  for (const label of ['Whisper-Medium ASR', 'Whisper-small', 'Qwen', 'FLAC', 'داده آموزشی', 'همه قطعه‌های پذیرفته‌شده', 'همراه همه داده‌های خط مبنا']) {
    assert.ok(html.includes(label), `Missing label: ${label}`);
  }
  assert.doesNotMatch(html, /<text\b[^>]*transform=/);
});

test('fusion overview preserves both views, shared encoder passes, attention, gate, and curriculum', () => {
  const html = readFileSync(dir + 'fusion-model-architecture-fa.html', 'utf8');
  for (const node of ['input', 'enhancer', 'enhanced-view', 'encoder', 'original-pass', 'enhanced-pass', 'fusion', 'cross-attention', 'gate', 'decoder', 'warmup', 'fusion-training', 'joint-training']) {
    assert.ok(html.includes(`data-node="${node}"`), `Missing component: ${node}`);
  }
  for (const label of ['U-Net', 'وزن‌های مشترک', 'توجه متقاطع', 'دوسویه', '۱ لایه و ۴ سر', 'دروازه زمان', 'همه بخش‌ها آموزش‌پذیر']) {
    assert.ok(html.includes(label), `Missing label: ${label}`);
  }
  const objectives = [...html.matchAll(/<text[^>]*class="training-loss ink latin"[^>]*>([\s\S]*?)<\/text>/g)]
    .map(match => match[1].replace(/<[^>]+>/g, '').trim());
  assert.deepEqual(objectives, ['0.5 Lenh + 0.5 Lfeat', 'LASR + 0.3 Lenh', 'LASR + 0.1 Lenh']);
  assert.equal((html.match(/۱ دوره؛ وزن‌های ویسپر ثابت/g) ?? []).length, 2);
  assert.doesNotMatch(html, /وزن‌های ویسپر آزاد|>نمای /);
  const enhancedView = html.split('data-node="enhanced-view"')[1].split('</g>')[0];
  assert.ok(enhancedView.indexOf('>لاگ‌مل</text>') < enhancedView.indexOf('>بهسازی‌شده</text>'));
  assert.ok(enhancedView.includes('بازنمایی باقیمانده'));
  for (const node of ['original-pass', 'enhanced-pass']) {
    assert.ok(html.split(`data-node="${node}"`)[1].split('</g>')[0].includes('>بازنمایی</text>'));
  }
  assert.match(html.split('data-node="joint-training"')[1], /class="training-note muted">۱ دوره<\/text>/);
  assert.match(html, /\.training-loss\s*\{[^}]*text-anchor:\s*end/);
  assert.doesNotMatch(html, /<text\b[^>]*transform=/);
});
