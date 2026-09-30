"""Download pinned model assets without executing remote model code."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
DETECTOR_ONLY={'yolox_tiny.onnx'}
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
    # sources.json is the embedding cache key (runtime.embeddings): changing its text re-embeds
    # every library. Detector files stay out of it so swapping a detector can't do that, and the
    # trailing newline matches the file already on disk, so a rerun keeps today's key.
    (ROOT/'models/sources.json').write_text(json.dumps(
        {k:v['url'] for k,v in manifest.items() if k not in DETECTOR_ONLY},indent=2)+'\n')

if __name__=='__main__':main()
