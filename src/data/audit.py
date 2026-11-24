"""Audit split counts, image readability, dimensions and exact duplicate leakage."""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
from src.config import SPLIT_ROOT, CLASS_NAMES

EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default=str(SPLIT_ROOT)); ap.add_argument('--output',default='outputs/data_audit.json'); a=ap.parse_args()
    root=Path(a.data_root); report={'data_root':str(root.resolve()),'splits':{},'exact_duplicate_hashes_across_splits':[],'warnings':[]}
    hash_splits=defaultdict(set); total=0
    for split in ('train','val','test'):
        report['splits'][split]={}; split_total=0
        for cls in CLASS_NAMES:
            folder=root/split/cls
            if not folder.exists(): raise FileNotFoundError(folder)
            files=sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in EXTS)
            bad=[]; dims=Counter()
            for f in files:
                try:
                    with Image.open(f) as im: im.verify()
                    with Image.open(f) as im: dims[f'{im.width}x{im.height}']+=1
                    hash_splits[digest(f)].add((split, cls, str(f)))
                except Exception as e: bad.append({'file':str(f),'error':str(e)})
            report['splits'][split][cls]={'image_count':len(files),'unreadable_count':len(bad),'common_dimensions':dims.most_common(5),'unreadable_examples':bad[:10]}
            split_total+=len(files)
        report['splits'][split]['TOTAL']=split_total; total+=split_total
    report['total_images']=total
    report['exact_duplicate_hashes_across_splits']=[{'sha256':h,'splits':sorted({item[0] for item in v}),'files':[{'split':item[0],'class':item[1],'path':item[2]} for item in sorted(v)]} for h,v in hash_splits.items() if len({item[0] for item in v})>1]
    if report['exact_duplicate_hashes_across_splits']: report['warnings'].append('Exact duplicate image content occurs across splits; investigate possible leakage.')
    report['warnings'].append('Filename/hash audit cannot establish patient-level separation. Patient identifiers are not available in this folder structure.')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
