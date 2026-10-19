#!/usr/bin/env node
/** Focused revision of the current Persian defense; preserves its OOXML assets. */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const here = path.dirname(fileURLToPath(import.meta.url));
const runtime = '/home/pouya/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill = '/home/pouya/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const sourceDefault = path.join(here, 'persian/robust_persian_asr_thesis_defense_fa_references_qwen_compact.pptx');
const help = `Revise the current 39-slide Persian defense into a 40-slide editable deck and PDF.

Usage: node Thesis/presentation/revise_persian_deck.mjs [options]

Options:
  --source PATH       Input PPTX (default: persian/robust_persian_asr_thesis_defense_fa_references_qwen_compact.pptx).
  --output-dir PATH   Deliverable directory (default: Thesis/presentation/persian).
  --basename NAME     Output stem (default: robust_persian_asr_thesis_defense_fa_reviewed).
  --build-dir PATH    Private staging directory (default: a new directory under /tmp).
  --pdf-renderer PATH LibreOffice executable (default: /usr/bin/soffice, stable renderer).
  --no-pdf            Export PPTX only (default: PPTX and PDF).
  -h, --help          Show help without loading authoring dependencies.

Adds units, English terminology, a portable slide 13, and native conclusion/contribution bullets.
Keeps source files, chart workbooks, references and appendix intact. Existing outputs are never overwritten.
See docs/script-guides/persian-slide-review.md for the workflow and validation.
`;
const args = process.argv.slice(2);
if (args.includes('--help') || args.includes('-h')) { process.stdout.write(help); process.exit(0); }
const options = { source: sourceDefault, outputDir: path.join(here, 'persian'), basename: 'robust_persian_asr_thesis_defense_fa_reviewed', pdf: true, pdfRenderer:'/usr/bin/soffice' };
const flags = { '--source': 'source', '--output-dir': 'outputDir', '--basename': 'basename', '--build-dir': 'buildDir', '--pdf-renderer':'pdfRenderer' };
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--no-pdf') { options.pdf = false; continue; }
  if (!flags[args[i]] || !args[i + 1] || args[i + 1].startsWith('--')) { console.error(`Invalid option or missing value: ${args[i]}\nUse --help.`); process.exit(2); }
  options[flags[args[i]]] = args[++i];
}
if (!/^[\w.-]+$/.test(options.basename)) { console.error('--basename must be a filename stem.'); process.exit(2); }

// Only edited paragraphs and newly authored slide parts are serialized. Original
// charts, embedded workbooks, media, references and complex equations stay intact.
const patchPackage = String.raw`
import sys, zipfile, re, posixpath, copy
from lxml import etree as E
src, authored, dst = sys.argv[1:]
A='http://schemas.openxmlformats.org/drawingml/2006/main'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PK='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
ns={'a':A,'p':P,'r':R}
def xml(b): return E.fromstring(b)
def emit(r): return E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
def value(r): return ''.join(r.xpath('.//a:t/text()',namespaces=ns))
def para_set(p,text,size=None):
    old=p.find('a:r/a:rPr',ns)
    for el in list(p):
        if E.QName(el).localname != 'pPr': p.remove(el)
    pr=p.find('a:pPr',ns)
    if pr is None: pr=E.SubElement(p,'{'+A+'}pPr')
    pr.set('rtl','1' if re.search('[\u0600-\u06ff]',text) else '0')
    # Separate Latin runs without invisible Unicode direction controls.
    for part in re.split(r'([A-Za-z][A-Za-z0-9 .,%()=+*/_-]*)',text):
        if not part: continue
        r=E.SubElement(p,'{'+A+'}r'); rp=copy.deepcopy(old) if old is not None else E.Element('{'+A+'}rPr')
        rp.set('lang','fa-IR' if re.search('[\u0600-\u06ff]',part) else 'en-US')
        if size: rp.set('sz',str(size))
        for kind in ['latin','ea','cs']:
            font=rp.find('a:'+kind,ns)
            if font is None: font=E.SubElement(rp,'{'+A+'}'+kind)
            font.set('typeface','Vazirmatn')
        r.append(rp); E.SubElement(r,'{'+A+'}t').text=part
def replace(root,old,new,size=None):
    matches=[p for p in root.findall('.//a:p',ns) if value(p)==old]
    assert matches, 'Missing paragraph: '+old
    for p in matches: para_set(p,new,size)
def subtitle(root,text):
    tree=root.find('p:cSld/p:spTree',ns)
    model=next(s for s in tree.findall('p:sp',ns) if value(s).strip())
    sp=copy.deepcopy(model)
    props=sp.find('p:nvSpPr/p:cNvPr',ns);props.set('id',str(max(int(p.get('id')) for p in root.findall('.//p:cNvPr',ns))+1));props.set('name','English terminology')
    xf=sp.find('p:spPr/a:xfrm',ns)
    xf.find('a:off',ns).attrib.update({'x':'685800','y':'1562100'})
    xf.find('a:ext',ns).attrib.update({'cx':'10820160','cy':'266700'})
    body=sp.find('p:txBody',ns)
    for p in body.findall('a:p',ns): body.remove(p)
    p=E.SubElement(body,'{'+A+'}p');para_set(p,text,1425)
    p.find('a:pPr',ns).set('algn','r')
    for rp in p.findall('a:r/a:rPr',ns):
        rp.set('b','0')
        fill=rp.find('a:solidFill',ns)
        if fill is None: fill=E.SubElement(rp,'{'+A+'}solidFill')
        for child in list(fill): fill.remove(child)
        E.SubElement(fill,'{'+A+'}srgbClr').set('val','52606d')
    tree.append(sp)
with zipfile.ZipFile(src) as z, zipfile.ZipFile(authored) as az:
    data={n:z.read(n) for n in z.namelist()}
    ct=xml(data['[Content_Types].xml']); act=xml(az.read('[Content_Types].xml'))
    types={el.get('PartName').lstrip('/'):el.get('ContentType') for el in act if el.get('PartName')}
    def addtype(name,kind):
        old=next((e for e in ct if e.get('PartName')=='/'+name),None)
        if old is None: E.SubElement(ct,'{'+CT+'}Override',PartName='/'+name,ContentType=kind)
    copied={}
    def transplant(part,target):
        if part in copied: return copied[part]
        copied[part]=target; content=az.read(part)
        rel=posixpath.join(posixpath.dirname(part),'_rels',posixpath.basename(part)+'.rels')
        if rel in az.namelist():
            rr=xml(az.read(rel))
            for link in rr:
                if link.get('TargetMode')=='External': continue
                dep=posixpath.normpath(posixpath.join(posixpath.dirname(part),link.get('Target'))).lstrip('/')
                if '/slideLayouts/' in dep: dest='ppt/slideLayouts/slideLayout1.xml'
                elif '/slides/' in dep and dep in copied: dest=copied[dep]
                else:
                    dest=posixpath.join(posixpath.dirname(dep),'review_'+posixpath.basename(dep))
                    transplant(dep,dest)
                link.set('Target',posixpath.relpath(dest,posixpath.dirname(target)))
            newrel=posixpath.join(posixpath.dirname(target),'_rels',posixpath.basename(target)+'.rels')
            data[newrel]=emit(rr)
        if part.endswith('.xml'):
            root=xml(content)
            for p in root.findall('.//a:p',ns):
                pr=p.find('a:pPr',ns)
                if pr is None: pr=E.Element('{'+A+'}pPr');p.insert(0,pr)
                pr.set('rtl','1' if re.search('[\u0600-\u06ff]',value(p)) else '0')
                for rp in p.findall('.//a:rPr',ns)+p.findall('a:endParaRPr',ns):
                    rp.set('lang','fa-IR' if re.search('[\u0600-\u06ff]',value(p)) else 'en-US')
                    if rp.find('a:cs',ns) is None: E.SubElement(rp,'{'+A+'}cs',typeface='Vazirmatn')
            content=emit(root)
        data[target]=content
        if part in types: addtype(target,types[part])
        return target
    for num,target in [(1,13),(2,23),(3,40)]: transplant(f'ppt/slides/slide{num}.xml',f'ppt/slides/slide{target}.xml')
    presentation=xml(data['ppt/presentation.xml']); ids=presentation.find('p:sldIdLst',ns)
    assert len(ids)==39, 'This revision expects the current 39-slide source.'
    rels=xml(data['ppt/_rels/presentation.xml.rels'])
    rid='rIdPersianReview'
    E.SubElement(rels,'{'+PK+'}Relationship',Id=rid,Type=R+'/slide',Target='slides/slide40.xml')
    item=E.Element('{'+P+'}sldId',id=str(max(int(e.get('id')) for e in ids)+1));item.set('{'+R+'}id',rid);ids.insert(23,item)
    data['ppt/presentation.xml']=emit(presentation);data['ppt/_rels/presentation.xml.rels']=emit(rels)
    # The first 24 slides are the main talk. References and appendices follow.
    for n in range(1,23):
        key=f'ppt/slides/slide{n}.xml';root=xml(data[key])
        for p in root.findall('.//a:p',ns):
            if re.fullmatch('[۰-۹]+ از ۲۳',value(p)): replace(root,value(p),value(p).replace('از ۲۳','از ۲۴'))
        data[key]=emit(root)
    replacements={
      6:[('اعتبارسنجی','اعتبارسنجی (نمونه)',1275),('آموزش','آموزش (نمونه)',1275),('تعدادها برای پیکره پایه‌اند، پیش از ساخت گونه‌های بلند و تخریب‌شده و افزودن ایران‌صدا','واحد شمارش جدول: نمونه گفتار (Utterance)، پیش از ساخت گونه‌های بلند و تخریب‌شده و افزودن ایران‌صدا',1425)],
      7:[('WER = (S + D + I) / N','WER (%) = 100 × (S + D + I) / N',2100),('نرخ خطای واژه','نرخ خطای واژه (٪)',1950),('CER بر پایه نویسه است. تجمیع با جمع شمارش خطاها و واحدهای مرجع انجام می‌شود','CER: Character Error Rate، نرخ خطای نویسه (٪)؛ تجمیع با جمع شمارش خطاها و واحدهای مرجع',1425)],
      10:[('نرخ خطای واژه در AGFarsdat','نرخ خطای واژه در AGFarsdat (٪)',2025)],
      11:[('حفظ بازنمایی اصلی در کنار بازنمایی بهسازی‌شده','حفظ دو بازنمایی با توجه متقاطع دوسویه (Bidirectional Cross-attention) و دروازه (Gate)',1500)],
      12:[('دروازه و ترکیب بازنمایی‌های اصلاح‌شده','دروازه (Gate) و ترکیب بازنمایی‌های اصلاح‌شده',1729),('LN: Layer Normalization    FFN: Feed-Forward Network    Heads: 4    FFN hidden dimension: 2D    Initial gate: 0.5','LN: Layer Normalization   FFN: Feed-Forward Network   4 heads   FFN: 2D features   Gate: 0.5 (unitless)',1200)],
      15:[('نرخ خطای واژه تجمیعی','نرخ خطای واژه تجمیعی (٪)',2025),('نرخ خطای واژه تماس واقعی','نرخ خطای واژه تماس واقعی (٪)',2025)],
      16:[('نرخ خطای واژه در AGFarsdat','نرخ خطای واژه در AGFarsdat (٪)',1950)],
      17:[('بازه ترجیحی ۱۵ تا ۲۵، سقف سخت ۲۸ ثانیه','بازه ترجیحی ۱۵ تا ۲۵ ثانیه، سقف سخت ۲۸ ثانیه',1870)],
      20:[('تغییر WER نسبت به مدل چندشرایطی، بر حسب درصد','تغییر WER نسبت به مدل چندشرایطی (واحد درصد)',1950)],
      21:[('WER تماس واقعی','WER تماس واقعی (٪)',1650),('WER تجمیعی','WER تجمیعی (٪)',1650)],
      32:[('AMR-NB (12.2k)','AMR-NB (12.2 kb/s)',1575),('AMR-WB (12.65k)','AMR-WB (12.65 kb/s)',1575)],
      36:[('ضریب مشترک و سقف دامنه ۰٫۹۹','ضریب مشترک و سقف دامنه ۰٫۹۹ نسبت به تمام‌مقیاس (بی‌واحد)',1425)],
      37:[('اعتبارسنجی','اعتبارسنجی (نمونه)',1575),('آموزش','آموزش (نمونه)',1575),('یک دوره، نرخ یادگیری ⁦1e-5⁩','یک دوره، نرخ یادگیری ۰٫۰۰۰۰۱ (بی‌واحد)',1650),('دسته ۴ و انباشت ۸ گام: ۳۲ نمونه','دسته ۴ نمونه، انباشت ۸ گام: ۳۲ نمونه',1650),('تعدادها مربوط به پیکره پایه‌اند. گونه‌های بلند، تخریب‌شده و ایران‌صدا در این جدول نیستند','واحد شمارش جدول: نمونه گفتار (Utterance). گونه‌های بلند، تخریب‌شده و ایران‌صدا در جدول نیستند',1425)],
      39:[('cyankiwi/Qwen3.6-27B-AWQ-INT4   ·   temperature = 0','cyankiwi/Qwen3.6-27B-AWQ-INT4   ·   temperature = 0 (unitless)',1280)]
    }
    glosses={
      2:'اعوجاج کدک (Codec Distortion) و افت بسته (Packet Loss) در تماس اینترنتی (VoIP)',
      3:'آموزش چندشرایطی (Multi-condition Training)، داده‌افزایی (Data Augmentation) و ریزتنظیم (Fine-tuning)',
      4:'بهسازی گفتار (Speech Enhancement) و همجوشی بازنمایی‌ها (Representation Fusion)',
      5:'خط مبنا (Baseline)، همجوشی دوجریانه (Dual-stream Fusion) و شبه‌برچسب‌گذاری (Pseudo-labeling)',
      7:'نرخ خطای واژه (Word Error Rate, WER)؛ اندازه هر مجموعه بر حسب نمونه گفتار (Utterance)',
      8:'نسبت سیگنال به نویز (Signal-to-Noise Ratio, SNR) و تغییر بهره (Gain)، هر دو بر حسب دسی‌بل (dB)',
      14:'پل تطبیقی (Adaptive Bridge): ترکیب آموخته‌شده صوت اصلی و بهسازی‌شده',
      16:'آزمایش حذف مؤلفه (Ablation Study): بررسی اثر نماها و توجه متقاطع (Cross-attention)',
      18:'رونویسی معلم (Teacher Transcription)، شبه‌برچسب (Pseudo-label) و پالایش کیفیت (Quality Filtering)',
      36:'هم‌ترازی زمانی (Time Alignment) و فراداده (Metadata) برای بازتولید آزمایش'
    }
    for n in set(replacements)|set(glosses):
        key=f'ppt/slides/slide{n}.xml';root=xml(data[key])
        for old,new,size in replacements.get(n,[]): replace(root,old,new,size)
        if n in glosses:
            subtitle(root,glosses[n])
            if n==36:
                added=next(sp for sp in root.findall('.//p:sp',ns) if sp.find('p:nvSpPr/p:cNvPr',ns).get('name')=='English terminology')
                added.find('p:spPr/a:xfrm/a:off',ns).set('y',str(538*9525))
        if n==16:
            # Keep the added glossary clear of the original diagram/chart label.
            original='نرخ خطای واژه در AGFarsdat (٪)'
            sp=next(s for s in root.findall('.//p:sp',ns) if value(s)==original)
            xf=sp.find('p:spPr/a:xfrm',ns)
            xf.find('a:off',ns).set('y',str(194*9525))
            xf.find('a:ext',ns).set('cy',str(35*9525))
        if n==7:
            rows=root.findall('.//a:tbl/a:tr',ns)
            for row in rows:
                p=row.find('a:tc/a:txBody/a:p',ns);para_set(p,value(p).strip()+' نمونه',1575)
        data[key]=emit(root)
    # Workbooks and values remain intact. Normalize pre-existing chart-label
    # child order so the revised package follows the Office XML schema.
    label_order={name:i for i,name in enumerate(['idx','delete','layout','tx','numFmt','spPr','txPr','dLblPos','showLegendKey','showVal','showCatName','showSerName','showPercent','showBubbleSize','separator','extLst'])}
    for key in list(data):
        if not key.startswith('ppt/') or not key.endswith('.xml'): continue
        root=xml(data[key]);changed=False
        if key.startswith('ppt/charts/chart'):
            # Source charts contain placeholder references ('categories', '0'),
            # complete caches, and no workbook/externalData. Reauthor that data
            # as honest literals, then let the finalizer create new workbooks.
            # These are new snapshots, never recovered source formula lineage.
            cns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
            for ref in root.findall('.//c:strRef',cns)+root.findall('.//c:numRef',cns):
                kind=E.QName(ref).localname;cache=ref.find('c:'+('strCache' if kind=='strRef' else 'numCache'),cns)
                assert cache is not None, 'Chart source cache is incomplete'
                count=cache.find('c:ptCount',cns)
                assert count is not None and int(count.get('val'))==len(cache.findall('c:pt',cns))
                formula=ref.find('c:f',cns)
                assert formula is not None and formula.text in ['categories','0'], 'Unexpected source formula lineage'
                literal=E.Element('{'+cns['c']+'}'+('strLit' if kind=='strRef' else 'numLit'))
                for child in cache: literal.append(copy.deepcopy(child))
                ref.getparent().replace(ref,literal);changed=True
        for rp in root.findall('.//a:rPr',ns)+root.findall('.//a:defRPr',ns)+root.findall('.//a:endParaRPr',ns):
            children=list(rp)
            # Put run fills before typefaces, leaving all content unchanged.
            fonts=[c for c in children if E.QName(c).localname in ['latin','ea','cs','sym']]
            for fill in [c for c in children if E.QName(c).localname in ['noFill','solidFill','gradFill','blipFill','pattFill','grpFill']]:
                if fonts and list(rp).index(fill)>list(rp).index(fonts[0]):
                    rp.remove(fill);rp.insert(list(rp).index(fonts[0]),fill);changed=True
        for label in root.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}dLbl'):
            children=list(label);ordered=sorted(children,key=lambda c:label_order.get(E.QName(c).localname,99))
            if ordered!=children:
                for c in children: label.remove(c)
                for c in ordered: label.append(c)
                changed=True
        if changed: data[key]=emit(root)
    app=xml(data['docProps/app.xml']);slide_count=app.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Slides')
    if slide_count is not None: slide_count.text='40'
    data['docProps/app.xml']=emit(app)
    # The source reference headings omit complex-script font metadata. Resolve
    # those inherited runs explicitly without changing their visible content.
    for n in [24,25,26]:
        key=f'ppt/slides/slide{n}.xml';root=xml(data[key])
        for rp in root.findall('.//a:rPr',ns)+root.findall('.//a:defRPr',ns):
            if rp.find('a:cs',ns) is None: E.SubElement(rp,'{'+A+'}cs',typeface='Vazirmatn')
        data[key]=emit(root)
    data['[Content_Types].xml']=emit(ct)
    with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as out:
        for name,content in data.items(): out.writestr(name,content)
`;

async function main() {
  process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
  const build = options.buildDir ? path.resolve(options.buildDir) : await fs.mkdtemp(path.join(os.tmpdir(), 'persian-review-'));
  await fs.mkdir(build, { recursive: true }); await fs.mkdir(options.outputDir, { recursive: true });
  const destination = path.resolve(options.outputDir, `${options.basename}.pptx`);
  const pdf = destination.replace(/\.pptx$/, '.pdf');
  for (const p of [destination, ...(options.pdf ? [pdf] : [])]) {
    if (await fs.stat(p).then(() => true).catch(() => false)) throw new Error(`Output exists: ${p}`);
  }
  const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtime, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
  const { makeNativeBulletParagraphs, finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
  const pres = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  pres.theme.defaultFont = 'Vazirmatn';
  function text(s, value, x, y, w, h, size=28, { bold=false, color='#1f2933', align='right', name=value }={}) {
    const shape=s.shapes.add({ geometry:'textbox', name, position:{left:x,top:y,width:w,height:h}, fill:'none', line:{fill:'none',width:0} });
    shape.text=value;shape.text.style={typeface:'Vazirmatn',fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',insets:0,wrap:'square',autoFit:'none'};return shape;
  }
  const fa=n=>String(n).replace(/\d/g,d=>'۰۱۲۳۴۵۶۷۸۹'[d]);
  function slide(title,num,dark=false) {
    const s=pres.slides.add();s.background.fill=dark?'#153d42':'#f7f6f2';
    text(s,dark?'نتیجه‌گیری و دستاوردهای پژوهش':'بهسازی و همجوشی',440,30,768,32,18,{bold:true,color:dark?'#b8d8db':'#1d6f8a'});
    text(s,title,72,75,1136,88,42,{bold:true,color:dark?'#f7f6f2':'#153d42'});
    // Existing footer geometry, matching the reference deck.
    s.shapes.add({geometry:'line',position:{left:72,top:674,width:1136,height:0},fill:'none',line:{fill:dark?'#436268':'#d5dcdd',width:1}});
    text(s,`${fa(num)} از ۲۴`,72,681,100,25,16,{align:'left',color:dark?'#b8d8db':'#7b8794'});return s;
  }
  const figure=path.join(build,'fusion-training.png');
  execFileSync(path.join(runtime,'python/bin/python3'),['-c','import zipfile,sys;from pathlib import Path;Path(sys.argv[2]).write_bytes(zipfile.ZipFile(sys.argv[1]).read("ppt/media/image5.png"))',options.source,figure],{stdio:'inherit'});
  const sourceBytes=await fs.readFile(figure);
  let s=slide('آموزش سه‌مرحله‌ای همجوشی',13);
  text(s,'گرم‌کردن بهساز (Enhancer Warm-up)، آموزش همجوشی (Fusion Training) و ریزتنظیم مشترک (Joint Fine-tuning)',72,164,1136,30,19,{color:'#52606d'});
  s.images.add({blob:sourceBytes.buffer.slice(sourceBytes.byteOffset,sourceBytes.byteOffset+sourceBytes.byteLength),contentType:'image/png',alt:'سه مرحله آموزش، هر مرحله یک دوره؛ ضرایب زیان بی‌واحدند',fit:'contain',position:{left:72,top:220,width:1136,height:173}});
  text(s,'مرحله سوم: آموزش همه اجزا، شامل ویسپر',72,412,1136,48,29,{bold:true,color:'#1d6f8a'});
  text(s,'هر مرحله: ۱ دوره کامل (Epoch)',710,474,498,40,27,{bold:true,color:'#153d42'});
  text(s,'نرخ یادگیری مرحله نهایی (بی‌واحد)',710,524,498,36,24);
  text(s,'ماژول‌های جدید: ۰٫۰۰۰۱؛ ویسپر: ۰٫۰۰۰۰۰۵',710,568,498,38,23);
  text(s,'داده مرحله نهایی',72,474,548,40,27,{bold:true,color:'#153d42'});
  text(s,'نمونه‌های تخریب‌شده و بلند\nزیرمجموعه تصادفی ۲۵٪ از نمونه‌های گفتار پاک',72,525,548,78,25);
  text(s,'ضرایب زیان بی‌واحدند. این آزمایش هم معماری و هم برنامه آموزش بازشناس را تغییر می‌دهد',72,632,1136,28,19,{color:'#52606d'});
  s.speakerNotes.text='هر مرحله یک دوره کامل است. نرخ یادگیری ماژول‌های جدید در مرحله نهایی 1e-4 و ویسپر 5e-6 است. نرخ‌ها و وزن‌های زیان بی‌واحدند.\nمنبع: Thesis/chapters/work.tex؛ تصویر اصلی fusion-training.png.';

  function bullets(s,items) {
    const box=text(s,'',90,212,1098,365,29,{color:'#f7f6f2',name:'native research bullets'});
    box.text=makeNativeBulletParagraphs(items,{marginLeftPoints:22,hangingPoints:11,spaceAfterPoints:23});
  }
  s=slide('نتیجه‌گیری پژوهش',23,true);
  bullets(s,[
    'آموزش چندشرایطی: کاهش WER تماس واقعی از ۳۵٫۶۳٪ به ۲۵٫۸۰٪، برابر ۹٫۸۳ واحد درصد.',
    'همجوشی دوجریانه: WER تماس واقعی ۲۵٫۳۷٪، با سود افزوده ۰٫۴۳ واحد درصد.',
    'ایران‌صدا، WER تجمیعی را از ۱۷٫۸۳٪ به ۱۷٫۲۲٪ کاهش داد؛ WER تماس واقعی به ۲۶٫۶۶٪ افزایش یافت.',
    'تناسب داده با کانال هدف، بیشترین سود را در این آزمایش‌ها داشت؛ بهبود تجمیعی به همه دامنه‌ها منتقل نشد.'
  ]);
  text(s,'اختلاف کوچک همجوشی، بدون تکرار آزمایش و آزمون معناداری، برتری آماری را ثابت نمی‌کند',72,610,1136,42,21,{color:'#b8d8db'});
  s.speakerNotes.text='نتیجه‌ها مربوط به مقایسه‌های اصلی ویسپر کوچک‌اند. افت WER بر حسب واحد درصد با کاهش نسبی تفاوت دارد. همجوشی و ایران‌صدا مسیرهای مستقل هستند.\nمنابع: Thesis/chapters/results.tex؛ Thesis/chapters/conclusions.tex.';
  s=slide('نوآوری‌ها و دستاوردهای پژوهش',24,true);
  bullets(s,[
    'طراحی خط لوله بازتولیدپذیر ساخت جفت‌های پاک و مخابراتی، با ثبت بذر و فراداده هر نمونه.',
    'ارزیابی انتقال آموزش چندشرایطی به تماس واقعی فارسی و فراهم‌کردن خط مبنایی مقاوم برای مقایسه روش‌ها.',
    'پیاده‌سازی همجوشی دوجریانه با بهساز باقیمانده، رمزگذار مشترک، توجه متقاطع دوسویه و دروازه زمان–ویژگی.',
    'ساخت خط لوله ایران‌صدا برای قطعه‌بندی، تولید و اصلاح شبه‌برچسب‌ها و پالایش کیفیت؛ ارزیابی سود آن به تفکیک دامنه.'
  ]);
  text(s,'سپاسگزارم؛ پرسش‌ها',72,613,1136,40,27,{bold:true,color:'#b8d8db'});
  s.speakerNotes.text='این موارد دستاوردهای پیاده‌سازی و ارزیابی این پایان‌نامه‌اند و ادعای ابداع مستقل اجزای شناخته‌شده نیستند.\nمنبع: Thesis/chapters/conclusions.tex، بخش دستاوردهای پژوهش.';
  const authored=path.join(build,'authored.pptx');await (await PresentationFile.exportPptx(pres)).save(authored);
  const candidate=path.join(build,'candidate.pptx');const final=path.join(build,'final',`${options.basename}.pptx`);await fs.mkdir(path.dirname(final),{recursive:true});
  execFileSync(path.join(runtime,'python/bin/python3'),['-c',patchPackage,options.source,authored,candidate],{stdio:'inherit'});
  const referenceDir=path.join(build,'source-reference'); await fs.mkdir(referenceDir,{recursive:true});
  execFileSync(path.join(runtime,'bin/override/soffice'),[`-env:UserInstallation=${pathToFileURL(path.join(build,'lo-reference')).href}`,'--headless','--convert-to','pdf','--outdir',referenceDir,options.source],{stdio:'inherit'});
  const referencePdf=path.join(referenceDir,path.basename(options.source).replace(/\.pptx$/,'.pdf'));
  await finalizePresentation({workspaceDir:build,candidatePath:candidate,finalPath:final,pythonExecutable:path.join(runtime,'python/bin/python3'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,6,7,21,29,30,31,32,33,34,35,36,37,38,39].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:40,requiredNativeChartOwnerSlides:[10,15,16,20],materializeLiteralChartWorkbooks:true,requiredNativeTableOwnerSlides:[3,4,6,7,21,29,30,31,32,33,34,35,36,37,38,39],fontPolicy:{basis:'reference',families:['Vazirmatn'],referencePath:referencePdf,referenceSha256:createHash('sha256').update(await fs.readFile(referencePdf)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
  await fs.copyFile(final,destination);
  // The bundled LibreOfficeDev 26.8 preview draws negative columns upward.
  // Stable LibreOffice 24.2 renders the unchanged signed data correctly.
  if(options.pdf) execFileSync(options.pdfRenderer,[`-env:UserInstallation=${pathToFileURL(path.join(build,'lo')).href}`,'--headless','--convert-to','pdf','--outdir',path.dirname(destination),destination],{stdio:'inherit',env:{...process.env,GSETTINGS_BACKEND:'memory'}});
  console.log(JSON.stringify({pptx:destination,...(options.pdf?{pdf}:{}),build}));
}
main().catch(e=>{console.error(e.stack??e);process.exitCode=1;});
