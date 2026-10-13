#!/usr/bin/env node
/** Reproducible, editable Persian thesis defense. See docs/script-guides/persian-thesis-presentation.md. */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');
const runtime = '/home/pouya/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const defaultSkill = '/home/pouya/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const help = `Build the 20-slide Persian thesis defense as editable PowerPoint and PDF.

Usage: node Thesis/presentation/build_persian_deck.mjs [options]

Options:
  --output-dir PATH   Deliverable directory (default: Thesis/presentation/persian).
  --build-dir PATH    Private staging directory (default: a new directory under /tmp).
  --basename NAME     Output stem (default: robust_persian_asr_thesis_defense_fa).
  --runtime-dir PATH  Bundled dependency directory (default: Codex primary runtime).
  --skill-dir PATH    Presentation skill directory (default: installed 26.1011.11113).
  --no-pdf           Export PPTX only (default: export PPTX and PDF).
  -h, --help         Show this help and exit without loading authoring dependencies.

Existing deliverables are never overwritten. Use a different basename for a revision.
Text uses the bundled Vazirmatn Regular and Bold fonts. PDF embeds the rendered fonts.
Implementation figures reuse the exact English-deck image crops without translation.
A font installation ZIP accompanies the output for PowerPoint on other computers.
Private staging retains the validation receipt and numbered slide previews.
Sources: the updated English deck and current thesis work/results/conclusions.
`;
const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) { process.stdout.write(help); process.exit(0); }
const opts = { outputDir: path.join(here, 'persian'), basename: 'robust_persian_asr_thesis_defense_fa', runtimeDir: runtime, skillDir: defaultSkill, pdf: true };
const flags = { '--output-dir': 'outputDir', '--build-dir': 'buildDir', '--basename': 'basename', '--runtime-dir': 'runtimeDir', '--skill-dir': 'skillDir' };
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--no-pdf') { opts.pdf = false; continue; }
  const key = flags[argv[i]];
  if (!key || !argv[i + 1] || argv[i + 1].startsWith('--')) { console.error(`Invalid option or missing value: ${argv[i]}\nUse --help for usage.`); process.exit(2); }
  opts[key] = argv[++i];
}
if (!/^[\w.-]+$/.test(opts.basename)) { console.error('--basename must be a filename stem without directory separators.'); process.exit(2); }

const F = 'Vazirmatn', T = 'Vazirmatn';
const cropDir = path.join(here, 'assets/english-implementation-crops');
const fontDir = path.join(here, 'assets/fonts/vazirmatn');
const C = { paper: '#f7f6f2', white: '#ffffff', ink: '#1f2933', deep: '#153d42', muted: '#52606d', soft: '#7b8794', rule: '#d5dcdd', accent: '#1d6f8a', tint: '#e8f3f6', input: '#edf1f3', output: '#eef4ef' };
const fa = value => String(value).replace(/\d/g, n => '۰۱۲۳۴۵۶۷۸۹'[n]).replace(/\./g, '٫').replace(/,/g, '٬');
const percent = value => `${fa(value.toFixed(2))}٪`;
const source = name => `Thesis/chapters/${name}.tex`;
let presentation, applyPresentationChartFont;

function txt(s, text, x, y, w, h, size = 28, { bold = false, color = C.ink, align = 'right', font = F, fill = 'none', name = text } = {}) {
  const o = s.shapes.add({ geometry: 'textbox', name, position: { left: x, top: y, width: w, height: h }, fill, line: { fill: 'none', width: 0 } });
  o.text = text;
  o.text.style = { fontSize: size, typeface: font, bold, color, alignment: align, verticalAlignment: 'middle', insets: 0, wrap: 'square', autoFit: 'none' };
  return o;
}
function rule(s, x, y, w, color = C.rule, weight = 1) {
  return s.shapes.add({ geometry: 'line', position: { left: x, top: y, width: w, height: 0 }, fill: 'none', line: { fill: color, width: weight } });
}
function node(s, text, x, y, w, h, accent = false, size = 25) {
  const o = s.shapes.add({ geometry: 'roundRect', name: text, position: { left: x, top: y, width: w, height: h }, fill: accent ? C.tint : C.white, borderRadius: 8, line: { fill: accent ? C.accent : C.rule, width: accent ? 2 : 1 } });
  o.text = text; o.text.style = { fontSize: size, typeface: F, bold: accent, color: accent ? C.accent : C.ink, alignment: 'center', verticalAlignment: 'middle', insets: 12, autoFit: 'none', wrap: 'square' };
  return o;
}
function connect(s, a, b, { from = 'left', to = 'right', kind = 'straight', dashed = false, arrow = true } = {}) {
  return s.shapes.connect(a, b, { fromSide: from, toSide: to, kind, line: { fill: C.muted, width: 1.7, style: dashed ? 'dashed' : 'solid' }, head: { type: 'none' }, tail: { type: arrow ? 'triangle' : 'none', width: 'sm', length: 'sm' } });
}
function base(title, section, { dark = false } = {}) {
  const s = presentation.slides.add(); s.background.fill = dark ? C.deep : C.paper;
  const color = dark ? C.paper : C.deep;
  txt(s, section, 440, 30, 768, 32, 18, { color: dark ? '#b8d8db' : C.accent, bold: true });
  txt(s, title, 72, 75, 1136, 88, 42, { bold: true, color, font: T });
  rule(s, 72, 674, 1136, dark ? '#436268' : C.rule);
  txt(s, `${fa(presentation.slides.items.length)} از ۲۰`, 72, 681, 100, 25, 16, { align: 'left', color: dark ? '#b8d8db' : C.soft });
  return s;
}
function caption(s, text, y = 620, h = 44) { txt(s, text, 72, y, 1136, h, 20, { color: C.muted }); }
function notes(s, text, refs, time) {
  s.speakerNotes.text = `${text}\n\nزمان پیشنهادی: ${time}\n\nمنابع:\n${refs.join('\n')}`;
}
async function image(s, filename, x, y, w, h, alt) {
  const bytes = await fs.readFile(filename);
  s.images.add({ blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), contentType: 'image/png', alt, fit: 'contain', position: { left: x, top: y, width: w, height: h } });
}
function bars(s, categories, values, x, y, w, h, { max = 40, min = 0, colors, points = false, labels = 23, gap = 90, axis = false } = {}) {
  const palette = colors ?? values.map((_, i) => i === values.length - 1 ? C.accent : ['#aab6bb', '#8da1a9', '#63828e'][i % 3]);
  const series = { name: points ? 'تغییر نرخ خطای واژه' : 'نرخ خطای واژه', values: points ? values : values.map(v => Number((v / 100).toFixed(4))), fill: C.accent,
    points: palette.map((fill, idx) => ({ idx, fill })),
    dataLabelOverrides: values.map((v, idx) => ({ idx, text: points ? `${v > 0 ? '+' : '−'}${fa(Math.abs(v).toFixed(2))}` : percent(v), showValue: false, showSeriesName: false, showCategoryName: false, position: 'outEnd', textStyle: { fontSize: 25, fill: C.deep, bold: true } })),
    valuesFormatCode: points ? '+0.00;-0.00;0' : '0.00%' };
  const ch = s.charts.add('bar', { position: { left: x, top: y, width: w, height: h }, categories, series: [series], hasLegend: false,
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: gap, varyColors: true },
    xAxis: { visible: true, tickLabelPosition: 'low', textStyle: { fontSize: labels, fill: C.deep }, line: { fill: points ? C.deep : C.rule, width: points ? 1.8 : 1 }, majorGridlines: null },
    yAxis: { visible: axis, min: points ? min : min / 100, max: points ? max : max / 100, majorUnit: points ? 2 : max / 400, numberFormatCode: points ? '0' : '0%', textStyle: { fontSize: 18, fill: C.muted }, majorGridlines: axis ? { fill: C.rule, width: .7 } : null, line: { fill: 'none', width: 0 } },
    dataLabels: { showValue: true, showLegendKey: false, showSeriesName: false, showCategoryName: false, position: 'outEnd', numberFormatCode: points ? '+0.00;-0.00;0' : '0.00%', textStyle: { fontSize: 25, fill: C.deep, bold: true } },
    chartFill: 'none', chartLine: { fill: 'none', width: 0 }, plotAreaFill: 'none', plotAreaLine: { fill: 'none', width: 0 } });
  applyPresentationChartFont(ch, { fontFamily: F }); return ch;
}

async function build() {
  // 01 — The official title and identity, without a speculative defense date.
  let s = base('', '', {});
  await image(s, path.join(here, 'assets/sharif-logo-presentation.png'), 72, 50, 132, 132, 'نشان اصلی دانشگاه صنعتی شریف');
  txt(s, 'دانشگاه صنعتی شریف', 500, 53, 708, 50, 29, { bold: true, color: C.deep });
  txt(s, 'دانشکده مهندسی کامپیوتر', 500, 106, 708, 40, 22, { color: C.muted });
  rule(s, 270, 182, 938);
  txt(s, 'بازشناسی مقاوم گفتار', 230, 244, 978, 115, 64, { bold: true, color: C.deep, font: T });
  txt(s, 'در زبان فارسی', 230, 365, 978, 115, 64, { bold: true, color: C.deep, font: T });
  txt(s, 'پویا فلاح سیچانی', 650, 550, 558, 54, 29, { bold: true, color: C.deep });
  txt(s, 'استاد راهنما: دکتر حسین صامتی', 650, 605, 558, 44, 23, { color: C.muted });
  txt(s, 'دفاع پایان‌نامه کارشناسی ارشد', 72, 565, 440, 60, 23, { color: C.muted, align: 'left' });
  notes(s, 'این پایان‌نامه مسئله بازشناسی گفتار فارسی در کانال‌های تلفنی و تماس اینترنتی را بررسی می‌کند. مسیرهای اصلی پژوهش، آموزش متناسب با تخریب مخابراتی، بهسازی و همجوشی دوجریانه، و افزودن گفتار فارسی شبه‌برچسب‌خورده است. تمرکز ارائه بر روش‌های پیاده‌سازی‌شده و شواهد تجربی آن‌هاست.', ['Thesis/front/info.tex', 'Thesis/front/abstract.tex', 'Typeface: Vazirmatn v33.003, Saber Rastikerdar, SIL OFL 1.1. https://github.com/rastikerdar/vazirmatn/tree/v33.003'], '۲۰ ثانیه');

  // 02 — Two recording conditions, one direct comparison.
  s = base('شکاف میان گفتار آموزشی و تماس واقعی', 'مسئله پژوهش');
  txt(s, 'گفتار در دسترس', 700, 220, 508, 60, 33, { bold: true, color: C.deep });
  txt(s, 'پیکره‌های عمومی فارسی\nقطعه‌های عمدتاً کوتاه\nسبک‌ها و شرایط ضبط متنوع', 700, 305, 508, 200, 29);
  const target = s.shapes.add({ geometry: 'rect', position: { left: 72, top: 205, width: 552, height: 340 }, fill: C.deep, line: { fill: 'none', width: 0 } });
  txt(s, 'شرایط تماس', 104, 225, 484, 60, 33, { bold: true, color: C.paper });
  txt(s, 'نویز و پهنای‌باند محدود\nاعوجاج کدک و افت بسته\nتلفن و تماس اینترنتی', 104, 310, 484, 200, 29, { color: C.paper });
  txt(s, 'هدف: کاهش خطای تماس واقعی با داده‌های موجود', 72, 566, 1136, 60, 29, { bold: true, color: C.accent });
  caption(s, 'گردآوری تماس رونویسی‌شده پرهزینه و حساس به حریم خصوصی است', 632, 30);
  notes(s, 'داده‌های فارسی سبک‌های مختلف گفتار و ضبط را پوشش می‌دهند، اما نماینده کامل کانال مخابراتی نیستند. در تماس واقعی چند تخریب می‌توانند هم‌زمان رخ دهند. داده تلفنی رونویسی‌شده نیز به‌سادگی در دسترس نیست. بنابراین از داده برچسب‌خورده عمومی و صوت بدون برچسب کمک می‌گیریم. معیار موفقیت، انتقال به تماس واقعی خارج از شبیه‌ساز است.', [source('introduction'), source('preliminaries')], '۴۵ ثانیه');

  // 03 — Editable scientific comparison graph, read from right to left.
  s = base('ساختار آزمایش‌ها و خط مبنا', 'طرح پژوهش');
  const ordinary = node(s, 'ویسپر کوچک معمولی\nگفتار پایه و نمونه‌های بلند', 958, 335, 250, 120, false, 23);
  const mct = node(s, 'ویسپر کوچک\nچندشرایطی', 595, 335, 285, 120, true, 29);
  txt(s, 'افزودن گفتار تخریب‌شده', 595, 469, 285, 55, 22, { color: C.muted, align: 'center' });
  const fusion = node(s, 'بهسازی و همجوشی', 90, 215, 300, 92);
  const bridge = node(s, 'پل تطبیقی\nروش مقایسه‌ای همجوشی', 90, 352, 300, 92, false, 23);
  const iran = node(s, 'افزودن ایران‌صدا\nحفظ ترکیب کامل داده‌ها', 90, 489, 300, 92, false, 23);
  connect(s, ordinary, mct); connect(s, mct, fusion, { kind: 'elbow' }); connect(s, mct, bridge, { dashed: true }); connect(s, mct, iran, { kind: 'elbow' });
  caption(s, 'همجوشی و ایران‌صدا دو مسیر مستقل‌اند. سامانه ترکیبی آن‌ها ارزیابی نشده است');
  notes(s, 'مدل معمولی، نمونه‌های بلند ساخته‌شده را از ابتدا دارد. افزودن داده تخریب‌شده، مدل چندشرایطی را می‌سازد و همین مدل مرجع آزمایش‌های بعدی است. یک مسیر معماری بهسازی و همجوشی را می‌افزاید و مسیر دیگر داده ایران‌صدا را. پل تطبیقی، مقایسه‌ای برای مسیر همجوشی است. هیچ نتیجه‌ای برای همجوشی همراه با ایران‌صدا در این پایان‌نامه گزارش نشده است.', [source('work')], '۵۰ ثانیه');

  // 04 — Data source names are retained as identifiers.
  s = base('داده‌های آموزش', 'داده و ارزیابی');
  txt(s, 'گفتار پایه فارسی', 690, 205, 518, 55, 32, { bold: true, color: C.deep });
  txt(s, 'Common Voice Persian 25\nFarsSpon  (زیرمجموعه محدود)\nFLEURS Persian', 690, 294, 518, 205, 28);
  txt(s, '۲ تا ۴', 72, 205, 504, 120, 78, { bold: true, color: C.accent });
  txt(s, 'قطعه برای ساخت هر نمونه بلند', 72, 331, 504, 70, 28, { bold: true, color: C.deep });
  txt(s, 'اتصال درون همان بخش داده\nبیش از ۵ و حداکثر ۲۰ ثانیه\nحضور در خط مبنای معمولی', 72, 419, 504, 160, 25);
  caption(s, 'صوت تک‌کاناله ۱۶ کیلوهرتزی و نرمال‌سازی مشترک متن فارسی');
  notes(s, 'ترکیب پایه از کامن‌وویس فارسی نسخه ۲۵، زیرمجموعه محدود فارس‌اسپان و فلورز فارسی تشکیل می‌شود. برای کاهش محدودیت ناشی از کوتاهی قطعه‌ها، دو تا چهار قطعه از یک بخش داده به هم متصل می‌شوند. نمونه نهایی بیش از پنج و حداکثر بیست ثانیه است. این نمونه‌های بلند به مدل معمولی نیز داده شده‌اند. آموزش چندشرایطی، داده تخریب‌شده را به همین بنیاد اضافه می‌کند.', [source('work')], '۴۰ ثانیه');

  // 05 — Real calls are the central evaluation target.
  s = base('ارزیابی روی تماس واقعی و گفتار عمومی', 'داده و ارزیابی');
  txt(s, '۱۰٬۰۴۴', 670, 192, 538, 140, 88, { bold: true, color: C.accent });
  txt(s, 'نمونه تماس واقعی در AGFarsdat', 670, 340, 538, 60, 29, { bold: true, color: C.deep });
  txt(s, 'تلفن و تماس اینترنتی\nمستقل از آموزش و انتخاب مدل', 670, 413, 538, 105, 25);
  txt(s, 'آزمون‌های گفتار عمومی', 72, 216, 500, 55, 29, { bold: true, color: C.deep });
  txt(s, 'Common Voice Persian 25\nFLEURS Persian\nPersianSpeech\nPersian Speech Corpus', 72, 289, 500, 220, 26);
  rule(s, 72, 540, 1136);
  txt(s, 'نرخ خطای واژه', 850, 552, 358, 52, 26, { bold: true, color: C.deep });
  txt(s, 'WER = (S + D + I) / N', 72, 552, 680, 52, 30, { align: 'left', color: C.deep });
  caption(s, 'CER بر پایه نویسه است. تجمیع با جمع شمارش خطاها و واحدهای مرجع انجام می‌شود', 608, 30);
  caption(s, 'رمزگشایی حریصانه مشترک است. PersianSpeech فقط ۲۴ نمونه دارد و واجدشرایط‌بودن کمی تغییر می‌کند', 641, 25);
  notes(s, 'AGFarsdat شامل ده هزار و چهل و چهار نمونه تماس واقعی است و در آموزش یا انتخاب مدل حضور ندارد. چهار مجموعه دیگر، تعمیم به گفتار عمومی را می‌سنجند. در WER، جانشینی و حذف و درج بر تعداد واژه‌های مرجع تقسیم می‌شوند. CER همین مفهوم را در سطح نویسه می‌سنجد. نتیجه تجمیعی میانگین درصدهای مجموعه‌ها نیست، بلکه از جمع خطاها و واحدهای مرجع محاسبه می‌شود. مقایسه‌های تک‌جریانه روی ۲۱٬۸۴۸ نمونه واجد شرایط مشترک‌اند، اما تعداد نمونه‌های Persian Speech Corpus بین بعضی سامانه‌ها متفاوت است.', [source('work'), source('results')], '۵۵ ثانیه');

  // 06 — Reuse the exact English preparation crop, with Persian narration around it.
  s = base('داده‌افزایی مخابراتیِ بازتولیدپذیر', 'مقاوم‌سازی با داده');
  txt(s, 'ساخت جفت‌های پاک و تخریب‌شده با رونویسی مشترک', 72, 174, 1136, 60, 28, { color: C.muted });
  await image(s, path.join(cropDir, 'degradation-preparation.png'), 72, 244, 1136, 227.2, 'Original English preparation and channel-conditioning crop');
  txt(s, 'نویز و تغییر بهره', 710, 486, 498, 54, 29, { bold: true, color: C.deep });
  txt(s, 'DEMAND     SNR: −5 … 15 dB     Gain: ±6 dB', 650, 551, 558, 45, 23, { align: 'right' });
  txt(s, 'نمایه‌های تلفن و تماس اینترنتی', 72, 486, 500, 54, 29, { bold: true, color: C.deep });
  txt(s, 'ثبت بذر تصادفی و فراداده هر جفت', 72, 551, 500, 45, 25);
  caption(s, 'وزن نمایه‌ها یک توزیع طراحی‌شده است و از آمار اپراتور به دست نیامده است');
  notes(s, 'برای هر نمونه، یک نمایه مخابراتی انتخاب می‌شود و نویز، بهره و پهنای‌باند تغییر می‌کنند. نویز DEMAND با نسبت سیگنال به نویز منفی پنج تا پانزده دسی‌بل و تغییر بهره تا شش دسی‌بل به کار می‌رود. پردازش واقعی کدک و شبکه در مرحله بعد است. ورودی تخریب‌شده و هدف پاک هم‌تراز می‌مانند. هدف بهساز پهنای‌باند مشترک با ورودی دارد تا شبکه برای بازسازی فرکانس‌های حذف‌شده جریمه نشود. بذر و تنظیمات هر جفت ثبت می‌شوند. وزن نمایه‌ها پوشش طراحی‌شده شرایط است، نه برآورد توزیع یک اپراتور.', [source('work'), 'Thesis/figs/degradation-pipeline-architecture.png', 'configs/speech_enhancement/degradation.yaml'], '۶۰ ثانیه');

  // 07 — Two distinct codec/network implementations, with no equivalence implied.
  s = base('مسیرهای کدک و جفت‌های هم‌تراز', 'مقاوم‌سازی با داده');
  await image(s, path.join(cropDir, 'degradation-codecs.png'), 122, 166, 1035, 450.15, 'Original English codec paths, alignment and peak-safe training-pair crop');
  caption(s, 'G.711، GSM، AMR-NB، AMR-WB و Opus. هدف پاک، پهنای‌باند متناظر با کانال را حفظ می‌کند', 622, 25);
  caption(s, 'Opus بومی افت بسته و پنهان‌سازی خطا را مدل می‌کند. در مسیرهای دیگر، حذف موج یک تقریب است', 648, 23);
  notes(s, 'دو مسیر پیاده‌سازی باید از هم تفکیک شوند. در Opus بومی، افت بسته و پنهان‌سازی خطای رمزگشا در مسیر واقعی کدک مدل می‌شوند. برای کدک‌های دیگر، رفت‌وبرگشت رمزگذاری و رمزگشایی با FFmpeg اجرا می‌شود و اختلال شبکه می‌تواند با حذف بخش‌هایی از موج تقریب زده شود. این تقریب معادل رفتار واقعی همه شبکه‌ها نیست. کدک‌های بررسی‌شده شامل جی هفتصد و یازده، جی‌اس‌ام، ای‌ام‌آر باریک‌باند، ای‌ام‌آر پهن‌باند و اپوس هستند. خروجی‌ها از نظر طول هم‌تراز و از نظر پهنای‌باند هدف سازگارند.', [source('work'), 'Thesis/figs/degradation-pipeline-architecture.png'], '۴۵ ثانیه');

  // 08 — Main robustness finding. All bars use a zero baseline.
  s = base('انتقال بهبود به تماس‌های واقعی', 'نتیجه داده‌افزایی مخابراتی');
  txt(s, 'نرخ خطای واژه در AGFarsdat', 450, 174, 758, 55, 27, { bold: true, color: C.deep });
  bars(s, ['معمولی', 'فقط نویز', 'pMCT', 'چندشرایطی\nمخابراتی'], [35.63, 32.67, 30.01, 25.80], 440, 240, 768, 340, { max: 42, labels: 23, gap: 100 });
  txt(s, '۹٫۸۳', 72, 222, 310, 130, 88, { bold: true, color: C.accent });
  txt(s, 'واحد درصد کاهش خطا\nدر تماس‌های واقعی', 72, 364, 310, 110, 29, { bold: true, color: C.deep });
  txt(s, '۲۷٫۶٪ کاهش نسبی', 72, 503, 310, 55, 26, { color: C.muted });
  caption(s, 'WER تجمیعی نیز از ۲۳٫۲۷٪ به ۱۷٫۸۳٪ کاهش یافت. مقدار کمتر بهتر است', 605, 34);
  caption(s, 'اثر جداگانه کدک و اختلال شبکه در آزمایش حذفی سنجیده نشده است', 643, 25);
  notes(s, 'نویز به‌تنهایی و pMCT هر دو نسبت به آموزش معمولی بهترند. مجموعه کامل تخریب‌های مخابراتی، کمترین خطای تماس واقعی را بین این چهار روش دارد. کاهش از ۳۵٫۶۳ به ۲۵٫۸۰ درصد برابر با ۹٫۸۳ واحد درصد یا حدود ۲۷٫۶ درصد کاهش نسبی است. این نتیجه روی تماس‌های واقعی مستقل از شبیه‌ساز به دست آمده و از انتقال یادگیری از تخریب مصنوعی به کانال واقعی پشتیبانی می‌کند. با این حال، چون هر اثر کدک یا شبکه جداگانه حذف نشده، نمی‌توان سهم مستقل هر تخریب را از این آزمایش تعیین کرد.', [source('results')], '۸۰ ثانیه');

  // 09 — Preserve the original implementation architecture crop in English.
  s = base('معماری بهسازی و همجوشی دوجریانه', 'بهسازی و همجوشی');
  txt(s, 'حفظ نمای اصلی در کنار نمای بهسازی‌شده', 72, 169, 1136, 55, 28, { color: C.muted });
  await image(s, path.join(cropDir, 'fusion-architecture.png'), 72, 211, 1136, 399.7, 'Original English dual-stream enhancement and fusion architecture crop');
  caption(s, 'شروع از ویسپر کوچک چندشرایطی است. بهساز در حوزه لاگ‌مل کار می‌کند و موج صوتی بازسازی نمی‌کند');
  notes(s, 'بهسازی ممکن است بخشی از اطلاعات مفید گفتار را نیز تضعیف کند، بنابراین نمای اصلی را نگه می‌داریم. یک U-Net باقیمانده، لاگ‌مل هشتادباندی را اصلاح می‌کند. دو نما با یک رمزگذار ویسپر و وزن‌های مشترک، در دو گذر پردازش می‌شوند. یک لایه توجه متقاطع دوسویه با چهار سر، اطلاعات را میان دو جریان مبادله می‌کند. سپس دروازه‌ای برای هر زمان و ویژگی، بازنمایی‌ها را ترکیب و به رمزگشا می‌دهد. این بهساز موج بازسازی نمی‌کند و مدل از نقطه وارسی چندشرایطی آغاز می‌شود. فرمول ترکیب: H_f = g × H_e′ + (1−g) × H_n′.', [source('work'), 'Thesis/figs/fusion-model-architecture-left-to-right.png'], '۸۰ ثانیه');

  // 10 — Preserve the English training crop with Persian context around it.
  s = base('آموزش سه‌مرحله‌ای همجوشی', 'بهسازی و همجوشی');
  await image(s, path.join(cropDir, 'fusion-training.png'), 72, 220, 1136, 172.87, 'Original English three-stage fusion training crop');
  txt(s, 'مرحله سوم: آموزش همه اجزا، شامل ویسپر', 72, 415, 1136, 55, 29, { bold: true, color: C.accent });
  txt(s, 'هر مرحله یک دوره کامل', 710, 486, 498, 45, 27, { bold: true, color: C.deep });
  txt(s, 'نرخ نهایی: ماژول‌های جدید \u20661e-4\u2069\nویسپر \u20665e-6\u2069', 710, 542, 498, 70, 25);
  txt(s, 'داده مرحله نهایی', 72, 486, 520, 45, 27, { bold: true, color: C.deep });
  txt(s, 'نمونه‌های تخریب‌شده و بلند\nزیرمجموعه تصادفی ۲۵٪ از گفتار پاک', 72, 542, 520, 70, 25);
  caption(s, 'این آزمایش هم معماری و هم برنامه آموزش بازشناس را تغییر می‌دهد', 635, 30);
  notes(s, 'در مرحله اول، تنها بهساز با ترکیبی برابر از زیان بازسازی و تطبیق ویژگی آموزش می‌بیند. در مرحله دوم، بهساز و همجوشی با هدف بازشناسی و وزن ۰٫۳ برای بهسازی، در برابر ویسپر ثابت آموزش می‌بینند. در مرحله نهایی همه اجزا آزاد می‌شوند و وزن زیان بهسازی به ۰٫۱ می‌رسد. هر مرحله یک دوره است. نرخ یادگیری ماژول‌های تازه ده به توان منفی چهار و ویسپر پنج ضرب در ده به توان منفی شش است. مرحله آخر گفتار تخریب‌شده و بلند و بیست و پنج درصد نمونه‌های پاک را دارد. این آزمایش هم معماری و هم میزان آموزش بازشناس را تغییر می‌دهد.', [source('work'), 'Thesis/figs/fusion-model-architecture-left-to-right.png'], '۶۰ ثانیه');

  // 11 — Preserve the licensed prior-work figure unmodified.
  s = base('پل تطبیقی پیش از بازشناسی', 'روش مقایسه‌ای');
  await image(s, path.join(here, 'assets/cui2025-bridge-figure1.png'), 72, 222, 786, 363, 'شکل ۱ مقاله کویی و همکاران: سامانه و پیش‌بینی ضریب پل');
  txt(s, 'پیاده‌سازی این پایان‌نامه', 908, 208, 300, 65, 27, { bold: true, color: C.deep });
  txt(s, 'بهساز FRCRN ثابت\nبازشناس ویسپر ثابت\nآموزش فقط ماژول پل', 908, 297, 300, 175, 25);
  txt(s, 'یک ضریب ترکیب\nبرای هر گفتار', 908, 493, 300, 100, 27, { bold: true, color: C.accent });
  txt(s, 'Cui et al., ICASSP 2025, Fig. 1 (CC BY 4.0)', 72, 592, 786, 29, 18, { align: 'left', color: C.muted });
  caption(s, 'اجرای مستقل فارسی است. تفاوت بهساز و برنامه آموزش، انتساب اثر به محل ادغام را محدود می‌کند', 635, 29);
  notes(s, 'این شکل اصلی مقاله کویی و همکاران است و روش مرجع را نشان می‌دهد. پل، موج اصلی و موج بهسازی‌شده را پیش از بازشناسی ترکیب می‌کند. در پیاده‌سازی مستقل این پایان‌نامه، FRCRN و ویسپر کوچک چندشرایطی ثابت‌اند و تنها پل آموزش می‌بیند. از دو نما یک ضریب برای هر گفتار پیش‌بینی می‌شود. هدف آموزش از کیفیت ادراکی و اطلاعات بازشناسی ازپیش‌محاسبه‌شده استفاده می‌کند. این اجرا بازتولید تأییدشده کد نویسندگان نیست. امتیازهای اسلاید بعد، نتایج فارسی همین پایان‌نامه‌اند و به نتایج CHiME-4 مقاله تعلق ندارند.', [source('work'), 'Thesis/presentation/assets/cui2025-bridge-figure1.source.md', 'https://doi.org/10.1109/ICASSP49660.2025.10889737', 'https://arxiv.org/html/2501.02452v1#S1.F1'], '۷۰ ثانیه');

  // 12 — Matching chart order and zero baselines keep small effects honest.
  s = base('سود افزوده همجوشی و پل تطبیقی', 'نتایج بهسازی و همجوشی');
  txt(s, 'نرخ خطای واژه تجمیعی', 72, 177, 530, 60, 29, { bold: true, color: C.deep, align: 'center' });
  txt(s, 'نرخ خطای واژه تماس واقعی', 678, 177, 530, 60, 29, { bold: true, color: C.deep, align: 'center' });
  const sys = ['چندشرایطی', 'پل تطبیقی', 'همجوشی'];
  const colors = ['#aab6bb', '#63828e', C.accent];
  bars(s, sys, [17.83, 17.74, 17.64], 72, 244, 530, 310, { max: 22, colors, labels: 22 });
  bars(s, sys, [25.80, 25.84, 25.37], 678, 244, 530, 310, { max: 32, colors, labels: 22 });
  txt(s, 'کاهش WER با همجوشی: ۰٫۱۹ واحد درصد تجمیعی و ۰٫۴۳ واحد درصد در تماس واقعی', 72, 575, 1136, 44, 25, { bold: true, color: C.accent });
  caption(s, 'CER تجمیعی به همین ترتیب: ۷٫۹۳٪، ۸٫۰۰٪ و ۷٫۸۵٪. اختلاف‌ها فاصله اطمینان ندارند', 631, 30);
  notes(s, 'همجوشی نرخ خطای واژه تجمیعی را از ۱۷٫۸۳ به ۱۷٫۶۴ و خطای تماس واقعی را از ۲۵٫۸۰ به ۲۵٫۳۷ درصد رسانده است. پل تطبیقی از نظر WER تجمیعی کمی بهتر از مرجع است، اما CER آن بیشتر می‌شود و خطای تماس واقعی تقریباً در سطح مرجع می‌ماند. اختلاف‌های همجوشی کوچک‌اند و بدون اجرای چندبذری و فاصله اطمینان گزارش می‌شوند. این مقایسه مربوط به دو سامانه کامل با بهساز و برنامه آموزش متفاوت است. میانگین دروازه نمای بهسازی‌شده ۰٫۶۷۸۵ و میانه ۰٫۶۹۳۰ است، اما به‌علت مبادله اطلاعات پیش از دروازه، این آمار سهم علّی مستقل هر نما را نشان نمی‌دهد.', [source('results'), source('work')], '۷۵ ثانیه');

  // 13 — The scale and inference-only controls are explicit.
  s = base('آزمایش حذفیِ دو نما و توجه متقاطع', 'مطالعه مؤلفه‌های همجوشی');
  txt(s, 'آزمایش مستقل\nویسپر بسیار کوچک', 72, 213, 295, 120, 30, { bold: true, color: C.deep });
  txt(s, 'آموزش فقط با کامن‌وویس', 72, 346, 295, 88, 24, { color: C.muted });
  txt(s, 'WER کامن‌وویس\nبه ترتیب نمودار', 72, 446, 295, 78, 23, { bold: true, color: C.deep });
  txt(s, '۵۷٫۸۲، ۶۲٫۳۵،\n۶۴٫۱۰، ۶۳٫۸۱ درصد', 72, 533, 295, 70, 23, { color: C.accent });
  txt(s, 'نرخ خطای واژه در AGFarsdat', 416, 174, 792, 55, 27, { bold: true, color: C.deep });
  bars(s, ['همجوشی\nدوسویه', 'فقط دروازه', 'فقط نمای\nاصلی¹', 'فقط نمای\nبهسازی‌شده¹'], [63.50, 65.88, 69.23, 69.69], 408, 246, 800, 342, { max: 83, colors: [C.accent, '#63828e', '#8da1a9', '#aab6bb'], labels: 21 });
  caption(s, '¹ حذف نما در زمان استنتاج از مدل همجوشی آموزش‌دیده، نه آموزش مستقل یک مدل تک‌جریانه', 614, 30);
  caption(s, 'بودجه آموزش و تعداد پارامترهای نابرابر، تعیین سهم مستقل مؤلفه‌ها را محدود می‌کند', 646, 24);
  notes(s, 'این آزمایش مستقل روی ویسپر بسیار کوچک و با آموزش فقط بر کامن‌وویس انجام شده است. همجوشی دوسویه در بین این پیکربندی‌ها، روی هر دو آزمون کمترین خطا را دارد. حذف توجه و استفاده از دروازه تنها، خطا را بالا می‌برد. استفاده از نمای اصلی یا بهسازی‌شده به‌تنهایی نیز ضعیف‌تر است. دو حالت تک‌نما در زمان استنتاج از نقطه وارسی همجوشی ساخته شده‌اند و کنترل تک‌جریانه مستقلِ آموزش‌دیده نیستند. بنابراین شواهد از استفاده هم‌زمان دو نما پشتیبانی می‌کنند، اما اثر آموزش اضافی و تفاوت ظرفیت را کاملاً جدا نمی‌کنند.', [source('results'), 'report/cv25-tiny-official/final-tests/summary.json'], '۶۵ ثانیه');

  // 14 — Acquired, selected and accepted hours are never conflated.
  s = base('گردآوری و قطعه‌بندی ایران‌صدا', 'افزایش داده فارسی');
  await image(s, path.join(cropDir, 'iranseda-source-segmentation.png'), 72, 181, 370, 436.69, 'Original English IranSeda source validation and VAD segmentation crop');
  txt(s, '۳٬۷۷۷ کتاب صوتی', 520, 210, 688, 80, 44, { bold: true, color: C.deep });
  txt(s, 'حدود ۶٬۳۰۰ ساعت صوت خام دریافت‌شده', 520, 308, 688, 55, 28);
  txt(s, '۲۵۰ ساعت منتخب', 520, 395, 688, 80, 44, { bold: true, color: C.accent });
  txt(s, 'طول هدف قطعه‌ها ۲۰ ثانیه است\nبازه ترجیحی ۱۵ تا ۲۵، سقف سخت ۲۸ ثانیه', 520, 495, 688, 100, 27);
  caption(s, 'مجموعه گردآوری‌شده از زیرمجموعه انتخاب‌شده برای آموزش بزرگ‌تر است');
  notes(s, 'مجموع دریافت‌شده شامل ۳۷۷۷ کتاب صوتی و حدود ۶۳۰۰ ساعت صوت خام است. به‌دلیل محدودیت محاسبات، ۲۵۰ ساعت برای پردازش نهایی انتخاب شد. این مقدار هنوز حجم پیکره پذیرفته‌شده نیست. منبع و درهم‌سازی و قابلیت رمزگشایی بررسی می‌شوند و صوت خام باقی می‌ماند. پس از تبدیل به صوت تک‌کاناله شانزده کیلوهرتزی، Silero VAD مرز قطعه‌ها را تا حد ممکن روی سکوت یا افت انرژی قرار می‌دهد. طول هدف بیست ثانیه، بازه ترجیحی پانزده تا بیست و پنج ثانیه و سقف سخت بیست و هشت ثانیه است. قطعه‌ها هم‌پوشان نیستند و شناسه پایدار دارند.', [source('work'), 'Thesis/figs/long-audio-data-creation-pipeline.png'], '۶۰ ثانیه');

  // 15 — The teacher and text refiner have distinct roles.
  s = base('رونویسی معلم، پالایش و آموزش', 'افزایش داده فارسی');
  const steps = [
    ['iranseda-teacher.png', '۱. رونویسی معلم', 72, 180],
    ['iranseda-normalization.png', '۲. نرمال‌سازی فارسی', 688, 180],
    ['iranseda-refinement.png', '۳. اصلاح با Qwen', 72, 384],
    ['iranseda-filtering.png', '۴. پالایش کیفیت', 688, 384]
  ];
  for (const [filename, label, x, y] of steps) {
    txt(s, label, x, y, 520, 42, 25, { bold: true, color: C.accent });
    await image(s, path.join(cropDir, filename), x, y + 45, 520, 158, `Original English crop: ${filename}`);
  }
  txt(s, 'افزودن داده پذیرفته‌شده به ترکیب کامل گفتار معمولی، بلند و تخریب‌شده', 72, 603, 1136, 35, 25, { bold: true, color: C.deep });
  caption(s, 'معماری ویسپر کوچک، تنظیمات اصلی و تعداد گام‌های آموزش ثابت‌اند', 646, 24);
  notes(s, 'ویسپر متوسط سازگارشده با فارسی، متن معلم را تولید می‌کند. متن خام نگهداری و سپس با نرمال‌ساز مشترک فارسی پردازش می‌شود. Qwen فقط اصلاح محدود متن را با استفاده از بافت انجام می‌دهد و مرحله پالایش، قطعه‌های نامعتبر را حذف می‌کند. همه نمونه‌های پذیرفته‌شده، بدون حذف اجزای خط مبنا به ترکیب اصلی، بلند و تخریب‌شده افزوده می‌شوند. مدل کوچک جدید، تعداد گام‌ها، نرخ یادگیری و تنظیمات اصلی مشابه مرجع چندشرایطی دارد. ویسپر متوسط هم معلم شبه‌برچسب و هم مرجع زمینه‌ای با ظرفیت بیشتر است.', [source('work'), 'Thesis/figs/long-audio-data-creation-pipeline.png'], '۵۰ ثانیه');

  // 16 — A genuine translated prompt rule, not an invented transcript example.
  s = base('داده پذیرفته‌شده و ملاحظات برچسب', 'افزایش داده فارسی');
  txt(s, '۳۹٬۴۵۴', 760, 193, 448, 126, 78, { bold: true, color: C.accent });
  txt(s, 'قطعه پذیرفته‌شده', 760, 321, 448, 55, 29, { bold: true, color: C.deep });
  txt(s, '۲۱۸٫۵۶۴ ساعت', 700, 396, 508, 86, 43, { bold: true, color: C.deep });
  txt(s, '۴۴٬۸۲۰ بررسی‌شده، ۵٬۳۶۶ ردشده\n۱۳۶ فایل منبع، تفکیک در سطح کتاب', 700, 508, 508, 85, 24);
  txt(s, 'ترجمه بخشی از دستور پالایش', 72, 205, 556, 50, 24, { color: C.muted });
  txt(s, '«همه نشانه‌های عددی را دقیقاً\nبا همان رقم‌ها، علامت‌ها،\nجداکننده‌ها و قالب حفظ کن.»', 72, 279, 556, 173, 30, { bold: true, color: C.deep });
  txt(s, 'Qwen فقط متن و بافت را دریافت می‌کند\nرد موارد نامطمئن یا تغییرات بیش از حد\nحفظ متن خام در کنار نسخه اصلاح‌شده', 72, 477, 556, 118, 24);
  caption(s, 'ممیزی انسانی کمّی برای دقت رونویسی انجام نشده است. ۴۰ شکست عملیاتی جداگانه ثبت شده‌اند', 632, 29);
  notes(s, 'از ۴۴۸۲۰ قطعه بررسی‌شده، ۳۹۴۵۴ قطعه پذیرفته و ۵۳۶۶ قطعه رد شده‌اند. صوت پذیرفته‌شده از ۱۳۶ فایل منبع و در مجموع ۲۱۸٫۵۶۴ ساعت است. چهل شکست عملیاتی جدا ثبت شده‌اند. دستور نمایش‌داده‌شده، ترجمه یک قاعده واقعی در پرامپت پایان‌نامه است. اعداد باید دقیقاً حفظ شوند. اصلاح واژه فقط در صورت پشتیبانی قوی متن و بافت مجاز است. Qwen به صوت دسترسی ندارد. تغییرات بیش از حد و عدم قطعیت باعث رد می‌شوند. تقسیم در سطح کتاب انجام می‌شود تا قطعه‌های نزدیک در آموزش و اعتبارسنجی پخش نشوند. این تدابیر جای ممیزی انسانی کیفیت را نمی‌گیرند.', [source('work'), source('conclusions')], '۶۰ ثانیه');

  // 17 — Signed native values and an explicit zero make the domain trade-off visible.
  s = base('اثر ایران‌صدا به دامنه گفتار وابسته است', 'نتیجه افزایش داده');
  txt(s, 'تغییر WER نسبت به مدل چندشرایطی، بر حسب واحد درصد', 72, 170, 1136, 60, 26, { color: C.deep });
  bars(s, ['AGFarsdat', 'Common Voice', 'FLEURS', 'PersianSpeech¹', 'Persian Speech\nCorpus'], [0.86, 0.35, -6.04, -8.39, -5.03], 72, 239, 1136, 337, { max: 2, min: -10, points: true, colors: ['#7b8794', '#aab6bb', C.accent, C.accent, C.accent], labels: 21, axis: true, gap: 120 });
  txt(s, 'مقدار منفی یعنی بهبود. تغییر = با ایران‌صدا منهای خط مبنای چندشرایطی', 72, 578, 1136, 45, 24, { color: C.accent, bold: true });
  caption(s, 'تجمیعی: WER از ۱۷٫۸۳٪ به ۱۷٫۲۲٪ و CER از ۷٫۹۳٪ به ۷٫۷۰٪ رسید', 625, 26);
  caption(s, '¹ فقط ۲۴ نمونه. اثر سبک، کانال، کیفیت برچسب و نسبت نمونه‌برداری از هم جدا نشده است', 650, 23);
  notes(s, 'افزودن ایران‌صدا خطای تجمیعی را کاهش می‌دهد، اما اثر یکنواختی ندارد. در فلورز و پیکره گفتار فارسی بهبود قابل توجه است. تغییر بزرگ PersianSpeech با توجه به تنها بیست و چهار نمونه، مقدماتی است. در مقابل، خطای AGFarsdat به اندازه ۰٫۸۶ واحد درصد و کامن‌وویس ۰٫۳۵ واحد درصد بیشتر می‌شود. CER همین دو مجموعه نیز به‌ترتیب ۰٫۳۴ و ۰٫۰۷ واحد درصد افزایش دارد. WER تجمیعی از ۱۷٫۸۳ به ۱۷٫۲۲ درصد، یعنی ۰٫۶۱ واحد درصد بهبود یافته است. آزمایش فعلی مشخص نمی‌کند سبک، کانال، برچسب یا نسبت اختلاط کدام‌یک این الگو را ایجاد کرده است.', [source('results')], '۷۵ ثانیه');

  // 18 — The only results table, native and editable, with two different minima.
  s = base('نتیجه آزمایش‌های اصلی ویسپر کوچک', 'جمع‌بندی شواهد');
  const vals = [
    ['WER تماس واقعی', 'WER تجمیعی', 'سامانه ویسپر کوچک'],
    ['۳۵٫۶۳٪', '۲۳٫۲۷٪', 'معمولی'],
    ['۲۵٫۸۰٪', '۱۷٫۸۳٪', 'چندشرایطی'],
    ['۲۵٫۳۷٪', '۱۷٫۶۴٪', 'همجوشی پیشنهادی'],
    ['۲۶٫۶۶٪', '۱۷٫۲۲٪', 'چندشرایطی و ایران‌صدا']
  ];
  const table = s.tables.add({ rows: 5, columns: 3, left: 72, top: 225, width: 1136, height: 310, columnWidths: [310, 310, 516], values: vals });
  table.borders.assign({ fill: C.rule, width: .8, style: 'solid' });
  for (let r = 0; r < 5; r++) for (let c = 0; c < 3; c++) {
    const cell = table.getCell(r, c);
    cell.fill = r === 0 ? C.deep : C.paper;
    cell.text.style = { typeface: F, fontSize: r === 0 ? 24 : 28, color: r === 0 ? C.paper : C.ink, bold: r === 0, alignment: c === 2 ? 'right' : 'center', verticalAlignment: 'middle', insets: 14, autoFit: 'none' };
  }
  for (const [r, c] of [[3, 0], [4, 1]]) { table.getCell(r, c).fill = C.tint; table.getCell(r, c).text.style = { typeface: F, fontSize: 29, bold: true, color: C.accent, alignment: 'center', verticalAlignment: 'middle', insets: 14 }; }
  txt(s, 'کمترین خطای تماس واقعی: همجوشی', 72, 565, 545, 55, 25, { bold: true, color: C.accent });
  txt(s, 'کمترین خطای تجمیعی: ایران‌صدا', 663, 565, 545, 55, 25, { bold: true, color: C.accent });
  caption(s, 'ویسپر متوسط: ۱۴٫۳۶٪ تجمیعی و ۲۱٫۹۹٪ تماس واقعی، با ظرفیت و آموزش متفاوت', 637, 28);
  notes(s, 'بزرگ‌ترین بهبود کنترل‌شده مقاومت مخابراتی با آموزش چندشرایطی به دست می‌آید. همجوشی نسبت به این مرجع، سود عددی کوچکی اضافه می‌کند. ایران‌صدا کمترین خطای تجمیعی را بین مدل‌های کوچک دارد، در حالی که همجوشی کمترین خطای تماس واقعی را ثبت می‌کند. بنابراین معیار انتخاب به دامنه هدف بستگی دارد. ویسپر متوسط ۱۴٫۳۶ درصد WER تجمیعی و ۲۱٫۹۹ درصد خطای تماس دارد، اما چون ظرفیت و برنامه آموزش آن متفاوت است، این مقایسه اثر خالص داده را نشان نمی‌دهد. پل در بخش قبل بررسی شد و اینجا دوباره وارد جدول نشده است.', [source('results'), source('conclusions')], '۶۰ ثانیه');

  // 19 — Actionable scientific limits, without a dashboard of cards.
  s = base('محدودیت‌ها و آزمایش‌های بعدی', 'بحث و ادامه پژوهش');
  const rows = [
    [231, 'اعتمادپذیری نتایج', 'اجرای چندبذری و نمونه‌های مشترک دقیق\nبرآورد فاصله اطمینان برای اختلاف‌های کوچک'],
    [367, 'سهم و هزینه همجوشی', 'کنترل با بودجه آموزش برابر و آزمایش حذفی در Small\nاندازه‌گیری زمان اجرا و حافظه'],
    [503, 'کیفیت و تناسب داده', 'ممیزی انسانی متن معلم و اصلاح‌های مدل زبانی\nآزمایش نسبت داده و گفتار نزدیک به تماس']
  ];
  for (let i = 0; i < rows.length; i++) {
    const [y, heading, body] = rows[i];
    txt(s, fa(i + 1), 1138, y, 70, 80, 49, { bold: true, color: C.accent });
    txt(s, heading, 745, y, 353, 80, 29, { bold: true, color: C.deep });
    txt(s, body, 72, y, 626, 88, 24);
    if (i < 2) rule(s, 72, y + 111, 1136);
  }
  caption(s, 'پوشش محدود شبیه‌ساز و خصوصی‌بودن آزمون تماس واقعی نیز دامنه شواهد را محدود می‌کنند', 635, 28);
  notes(s, 'برای اختلاف‌های کوچک باید چند بذر اجرا شود و با نمونه‌های دقیقاً مشترک، فاصله اطمینان به دست آید. برای همجوشی، کنترل تک‌جریانه با آموزش اضافی برابر، آزمایش حذفی در مقیاس مدل کوچک و سنجش هزینه زمان و حافظه لازم است. کیفیت متن معلم و تغییرات Qwen باید جداگانه با نمونه رونویسی انسانی ممیزی شوند. برای ایران‌صدا نیز نسبت‌های مختلف اختلاط و داده نزدیک‌تر به تماس باید آزموده شوند. پوشش شبیه‌ساز کامل نیست و خصوصی‌بودن AGFarsdat، بازبینی مستقل بیرونی را محدود می‌کند.', [source('results'), source('conclusions')], '۵۰ ثانیه');

  // 20 — Keep the answer and the central numbers visible during questions.
  s = base('جمع‌بندی', 'پایان ارائه', { dark: true });
  txt(s, 'بیشترین بهبود با آموزش متناسب با کانال', 72, 167, 1136, 70, 35, { bold: true, color: C.paper });
  txt(s, '۳۵٫۶۳٪', 829, 259, 379, 126, 74, { bold: true, color: '#b8d8db' });
  txt(s, 'به', 735, 278, 60, 90, 32, { align: 'center', color: C.paper });
  txt(s, '۲۵٫۸۰٪', 322, 259, 379, 126, 74, { bold: true, color: C.paper });
  txt(s, 'نرخ خطای واژه در تماس‌های واقعی مستقل', 322, 391, 886, 55, 27, { color: '#b8d8db' });
  rule(s, 322, 474, 886, '#436268');
  txt(s, 'همجوشی', 856, 491, 352, 52, 28, { bold: true, color: C.paper });
  txt(s, '۲۵٫۳۷٪ خطای تماس واقعی\nبهبود عددی کوچک', 856, 550, 352, 82, 24, { color: '#b8d8db' });
  txt(s, 'داده کتاب صوتی', 376, 491, 410, 52, 28, { bold: true, color: C.paper });
  txt(s, '۱۷٫۲۲٪ خطای تجمیعی\nسود وابسته به دامنه', 376, 550, 410, 82, 24, { color: '#b8d8db' });
  txt(s, 'سپاسگزارم\nپرسش‌ها', 72, 532, 214, 95, 30, { color: C.paper, bold: true, align: 'left' });
  notes(s, 'آموزش متناسب با تخریب‌های مخابراتی، بیشترین بهبود مقاومت بازشناسی فارسی را در آزمایش‌های اصلی ایجاد کرد. نرخ خطای تماس واقعی از ۳۵٫۶۳ به ۲۵٫۸۰ درصد رسید. همجوشی سود عددی کوچک‌تری اضافه کرد و به ۲۵٫۳۷ درصد رسید. داده کتاب صوتی نتیجه تجمیعی را به ۱۷٫۲۲ درصد رساند، اما سود آن به دامنه وابسته بود و به تماس واقعی منتقل نشد. این پژوهش سامانه‌های بازتولیدپذیر و شواهدی برای سازگارسازی بازشناس فارسی در شرایط محدودیت داده مخابراتی فراهم می‌کند. سپاسگزارم و آماده پاسخ‌گویی به پرسش‌ها هستم.', [source('results'), source('conclusions')], '۴۰ ثانیه');
}

// Exporter language defaults do not encode Persian paragraph direction. Normalize
// OOXML metadata, never visual-order text: all text stays logical and editable.
const normalizeRtl = String.raw`
import sys, re, zipfile
from lxml import etree as E
src, dst = sys.argv[1:3]
A='http://schemas.openxmlformats.org/drawingml/2006/main'
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
NS={'a':A,'c':C}
def add_font(pr, kind, family):
    elem=pr.find('a:'+kind,NS)
    if elem is None: elem=E.SubElement(pr,'{'+A+'}'+kind)
    elem.set('typeface',family)
with zipfile.ZipFile(src) as z, zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as w:
    for info in z.infolist():
        data=z.read(info.filename)
        if info.filename.endswith('.xml') and info.filename.startswith(('ppt/slides/','ppt/notesSlides/','ppt/charts/','ppt/theme/')):
            root=E.fromstring(data)
            for para in root.findall('.//a:p',NS):
                text=''.join(para.findall('.//a:t',NS)[i].text or '' for i in range(len(para.findall('.//a:t',NS))))
                # Native chart default paragraphs have no text but govern categories.
                arabic=bool(re.search('[\u0600-\u06ff]',text))
                chart='/charts/' in info.filename
                pr=para.find('a:pPr',NS)
                if pr is None: pr=E.Element('{'+A+'}pPr'); para.insert(0,pr)
                pr.set('rtl','1' if arabic or chart else '0')
                if not text.strip(): continue
                for rp in para.findall('.//a:rPr',NS)+para.findall('.//a:defRPr',NS)+para.findall('.//a:endParaRPr',NS):
                    rp.set('lang','fa-IR' if arabic else 'en-US')
                    lat=rp.find('a:latin',NS)
                    family=lat.get('typeface','Vazirmatn') if lat is not None else 'Vazirmatn'
                    add_font(rp,'cs',family)
            # LibreOffice otherwise treats omitted per-series/per-point flags as
            # true even when the chart-level legend-key flag is false.
            for label in root.findall('.//c:dLbls',NS)+root.findall('.//c:dLbl',NS):
                flag=label.find('c:showLegendKey',NS)
                if flag is None:
                    flag=E.Element('{'+C+'}showLegendKey')
                    following=next((el for el in label if E.QName(el).localname in ('showVal','showCatName','showSerName','showPercent','showBubbleSize','separator','showLeaderLines','leaderLines','extLst')),None)
                    if following is None: label.append(flag)
                    else: label.insert(label.index(following),flag)
                flag.set('val','0')
            for scheme in root.findall('.//a:fontScheme',NS):
                for family_tag in ('majorFont','minorFont'):
                    family=scheme.find('a:'+family_tag,NS)
                    if family is not None: add_font(family,'cs','Vazirmatn')
            data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        w.writestr(info,data)
`;

async function main() {
  opts.outputDir = path.resolve(opts.outputDir);
  opts.buildDir = opts.buildDir ? path.resolve(opts.buildDir) : await fs.mkdtemp(path.join(os.tmpdir(), 'ms-thesis-persian-'));
  await fs.mkdir(opts.buildDir, { recursive: true });
  await fs.mkdir(opts.outputDir, { recursive: true });
  const pptxDestination = path.join(opts.outputDir, `${opts.basename}.pptx`);
  const pdfDestination = path.join(opts.outputDir, `${opts.basename}.pdf`);
  const fontsDestination = path.join(opts.outputDir, `${opts.basename}_fonts.zip`);
  for (const filename of [pptxDestination, fontsDestination, ...(opts.pdf ? [pdfDestination] : [])]) {
    if (await fs.stat(filename).then(() => true).catch(() => false)) throw new Error(`Output exists: ${filename}. Choose --basename for a revision.`);
  }
  // Register bundled font files for this build without changing system fonts.
  const fontCache = path.join(opts.buildDir, 'font-cache');
  await fs.mkdir(fontCache, { recursive: true });
  const fontConfig = path.join(opts.buildDir, 'fonts.conf');
  const escapeXml = value => value.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
  await fs.writeFile(fontConfig, `<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig><include>/etc/fonts/fonts.conf</include><dir>${escapeXml(fontDir)}</dir><cachedir>${escapeXml(fontCache)}</cachedir></fontconfig>\n`);
  process.env.FONTCONFIG_FILE = fontConfig;
  process.env.RUNTIME_NODE_MODULES = path.join(opts.runtimeDir, 'node/node_modules');
  const modulePath = path.join(process.env.RUNTIME_NODE_MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs');
  const { Presentation, PresentationFile } = await import(pathToFileURL(modulePath).href);
  const utils = await import(pathToFileURL(path.join(opts.skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
  applyPresentationChartFont = utils.applyPresentationChartFont;
  presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  presentation.theme.defaultFont = F;
  await build();
  const staging = path.join(opts.buildDir, 'staging');
  const finalDir = path.join(opts.buildDir, 'final');
  await fs.mkdir(staging, { recursive: true }); await fs.mkdir(finalDir, { recursive: true });
  const draft = path.join(staging, 'draft.pptx'), candidate = path.join(staging, 'candidate-rtl.pptx');
  await (await PresentationFile.exportPptx(presentation)).save(draft);
  const python = path.join(opts.runtimeDir, 'python/bin/python3');
  execFileSync(python, ['-c', normalizeRtl, draft, candidate], { stdio: 'inherit' });
  const final = path.join(finalDir, `${opts.basename}.pptx`);
  await utils.finalizePresentation({ workspaceDir: opts.buildDir, candidatePath: candidate, finalPath: final,
    pythonExecutable: python,
    integrityValidatorPath: path.join(opts.skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath: path.join(opts.skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '18'],
    explicitTotalSlideCount: 20,
    requiredNativeChartOwnerSlides: [8, 12, 13, 17], requiredNativeTableOwnerSlides: [18],
    materializeLiteralChartWorkbooks: true,
    fontPolicy: { basis: 'design', families: [F] }, verifyArtifactToolImport: true,
    receiptPath: path.join(staging, 'validation.json') });
  await fs.copyFile(final, pptxDestination);
  if (opts.pdf) {
    const soffice = path.join(opts.runtimeDir, 'bin/override/soffice');
    const profile = pathToFileURL(path.join(opts.buildDir, 'libreoffice-profile')).href;
    execFileSync(soffice, [`-env:UserInstallation=${profile}`, '--headless', '--convert-to', 'pdf', '--outdir', finalDir, final], { stdio: 'inherit', env: { ...process.env, GSETTINGS_BACKEND: 'memory' } });
    await fs.copyFile(path.join(finalDir, `${opts.basename}.pdf`), pdfDestination);
    const previews = path.join(opts.buildDir, 'previews'); await fs.mkdir(previews, { recursive: true });
    execFileSync(path.join(opts.runtimeDir, 'bin/override/pdftoppm'), ['-scale-to', '1600', '-png', pdfDestination, path.join(previews, 'slide')], { stdio: 'inherit' });
  }
  execFileSync(python, ['-c', 'from pathlib import Path; from zipfile import ZipFile, ZIP_DEFLATED; import sys; src=Path(sys.argv[1]); z=ZipFile(sys.argv[2], "w", ZIP_DEFLATED); [(z.write(p,p.name)) for p in sorted(src.iterdir()) if p.is_file()]; z.close()', fontDir, fontsDestination], { stdio: 'inherit' });
  console.log(`PowerPoint: ${pptxDestination}\n${opts.pdf ? `PDF: ${pdfDestination}\n` : ''}Fonts: ${fontsDestination}\nPrivate build: ${opts.buildDir}`);
}
main().catch(error => { console.error(error.stack ?? error.message); process.exitCode = 1; });
