#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil,tarfile,zipfile,subprocess,sys,os
R=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data')
# Rebuild validation report; source failures stop packaging. HDL BLOCKED is recorded, not hidden.
subprocess.run([sys.executable,'tools/build_validation_report.py'],cwd=R,check=True)
exclude={'build','.git','__pycache__'}
def included(p:Path):return p.is_file() and not any(x in exclude for x in p.relative_to(R).parts) and p.name not in {'MANIFEST.sha256','MANIFEST.json'}
files=sorted(p for p in R.rglob('*') if included(p))
manifest=[]
for p in files:
    data=p.read_bytes();manifest.append({'path':p.relative_to(R).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
(R/'MANIFEST.json').write_text(json.dumps({'release':'FusionX-D2','file_count':len(manifest),'files':manifest},indent=2)+'\n')
(R/'MANIFEST.sha256').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in manifest))
files=sorted(p for p in R.rglob('*') if included(p) or p.name in {'MANIFEST.json','MANIFEST.sha256'})
zip_path=OUT/'FusionX_D2_Validated_Source_Design_DB.zip';tgz_path=OUT/'FusionX_D2_Validated_Source_Design_DB.tar.gz'
for x in (zip_path,tgz_path):
    if x.exists():x.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:z.write(p,arcname=f'FusionX_D2_Validated_Source_Design_DB/{p.relative_to(R).as_posix()}')
with tarfile.open(tgz_path,'w:gz') as t:
    for p in files:t.add(p,arcname=f'FusionX_D2_Validated_Source_Design_DB/{p.relative_to(R).as_posix()}')
# Verify extraction and internal manifest.
check=OUT/'_fusionx_d2_archive_check';shutil.rmtree(check,ignore_errors=True);check.mkdir()
with zipfile.ZipFile(zip_path) as z:
    bad=z.testzip();z.extractall(check)
if bad:raise SystemExit(f'bad zip member {bad}')
E=check/'FusionX_D2_Validated_Source_Design_DB';m=json.loads((E/'MANIFEST.json').read_text());bad_files=[]
for x in m['files']:
    p=E/x['path']
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:bad_files.append(x['path'])
if bad_files:raise SystemExit(f'manifest mismatch {bad_files[:5]}')
result={'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'tar_gz':str(tgz_path),'tar_gz_bytes':tgz_path.stat().st_size,'tar_gz_sha256':hashlib.sha256(tgz_path.read_bytes()).hexdigest(),'internal_files':m['file_count'],'archive_integrity_pass':True}
(OUT/'FusionX_D2_Archive_Verification.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'FusionX_D2_Validated_Source_Design_DB.zip.sha256').write_text(result['zip_sha256']+'  FusionX_D2_Validated_Source_Design_DB.zip\n')
for src,dst in [('FINAL_VALIDATION_REPORT.md','FusionX_D2_Final_Validation_Report.md'),('STATUS.md','FusionX_D2_Status.md'),('docs/07_AUDIT_CLOSURE.md','FusionX_D2_Audit_Closure.md'),('reports/FINAL_VERIFICATION_SUMMARY.json','FusionX_D2_Verification_Summary.json'),('foundry/N3P_HANDOFF_REQUIREMENTS.yaml','FusionX_D2_N3P_Handoff_Requirements.yaml')]:shutil.copy2(R/src,OUT/dst)
print(json.dumps(result,indent=2))
