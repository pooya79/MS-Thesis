#!/usr/bin/env node
// Render the editable, offline HTML sources to vector PDFs for XeLaTeX.
import { parseArgs } from 'node:util';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { mkdir, readdir, writeFile } from 'node:fs/promises';

const { values } = parseArgs({ options: {
  help: { type: 'boolean', short: 'h' },
  'source-dir': { type: 'string', default: dirname(fileURLToPath(import.meta.url)) },
  'output-dir': { type: 'string' },
  'preview-dir': { type: 'string' },
  'browser-path': { type: 'string', default: '/usr/bin/google-chrome' },
  'module-dir': { type: 'string' },
} });
if (values.help) {
  console.log(`Export Persian thesis figures as single-page vector PDFs.

Usage: node Thesis/figs/export_persian_figures.mjs [options]

--source-dir PATH    HTML directory (default: this script's directory)
--output-dir PATH    PDF directory (default: source directory; overwrites exports)
--preview-dir PATH   Optional PNG previews and geometry report (default: disabled)
--browser-path PATH Chromium executable (default: /usr/bin/google-chrome)
--module-dir PATH   Node modules directory containing playwright (default: Node resolution)
--help, -h          Show help without loading Playwright or writing files

Reads only *-fa.html. Embedded fonts make the figures independent of network access.
Geometry checks reject clipped text and labels outside their own nodes.`);
  process.exit(0);
}
const require = createRequire(import.meta.url);
const { chromium } = require(values['module-dir'] ? join(resolve(values['module-dir']), 'playwright') : 'playwright');
const sourceDir = resolve(values['source-dir']);
const outputDir = resolve(values['output-dir'] ?? sourceDir);
const files = (await readdir(sourceDir)).filter(name => name.endsWith('-fa.html')).sort();
if (!files.length) throw new Error(`No *-fa.html sources in ${sourceDir}`);
await mkdir(outputDir, { recursive: true });
if (values['preview-dir']) await mkdir(values['preview-dir'], { recursive: true });
const browser = await chromium.launch({ executablePath: values['browser-path'], headless: true });
const reports = [];
try {
  const page = await browser.newPage({ viewport: { width: 1120, height: 792 }, deviceScaleFactor: 2 });
  await page.route(/^https?:/, route => route.abort());
  for (const name of files) {
    await page.goto(pathToFileURL(join(sourceDir, name)).href);
    await page.evaluate(() => document.fonts.ready);
    const report = await page.evaluate(() => {
      const issues = [];
      const svg = document.querySelector('svg');
      const view = svg.viewBox.baseVal;
      const contains = (outer, inner, padding = 0) => inner.x >= outer.x + padding - 1 &&
        inner.y >= outer.y + padding - 1 && inner.x + inner.width <= outer.x + outer.width - padding + 1 &&
        inner.y + inner.height <= outer.y + outer.height - padding + 1;
      for (const text of svg.querySelectorAll('text')) {
        const bounds = text.getBBox();
        if (!contains(view, bounds)) issues.push(`Outside figure: ${text.textContent}`);
        const node = text.closest('[data-node]');
        if (node && !contains(node.querySelector('rect').getBBox(), bounds, 4)) {
          issues.push(`Outside node padding: ${text.textContent}`);
        }
      }
      return { issues, fontLoaded: document.fonts.check('24px Vazirmatn'), width: view.width, height: view.height };
    });
    reports.push({ file: name, ...report });
    if (report.issues.length || !report.fontLoaded) {
      console.error(JSON.stringify(reports.at(-1), null, 2));
      continue;
    }
    await page.pdf({ path: join(outputDir, name.replace('.html', '.pdf')), preferCSSPageSize: true, printBackground: true });
    if (values['preview-dir']) await page.screenshot({ path: join(values['preview-dir'], name.replace('.html', '.png')) });
    console.log(`Exported ${name.replace('.html', '.pdf')}`);
  }
} finally {
  await browser.close();
  if (values['preview-dir']) await writeFile(join(values['preview-dir'], 'geometry.json'), JSON.stringify(reports, null, 2));
}

if (reports.some(report => report.issues.length || !report.fontLoaded)) process.exitCode = 1;
