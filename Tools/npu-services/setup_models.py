"""Download pinned model assets without executing remote model code."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'model-manifest.json').read_text())
    for name,item in manifest.items():
        path=ROOT/'models'/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.download')
        urllib.request.urlretrieve(item['url'],temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Checksum mismatch: '+name)
        temporary.replace(path)
    (ROOT/'models/sources.json').write_text(json.dumps({k:v['url'] for k,v in manifest.items()},indent=2))

if __name__=='__main__':main()
