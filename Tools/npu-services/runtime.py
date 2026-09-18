"""Shared local inference on fixed devices. No AUTO, no fallback: each model names the
device it runs on, and a missing device is an error rather than a silent move to the CPU.

Text embedding runs on the Intel iGPU, picked by measurement (lecture-notes
tools/semantic_search_benchmark.py, 2026-09-18, 6155 passages of real notes/transcripts):

                          index       CPU time   per passage   compile
  Intel iGPU, batch 32     35.2s        11.1s        5.7 ms      4.3s
  NPU, one at a time       55.6s        13.8s        9.0 ms      7.4s
  CPU, one at a time      199.5s      1022.3s       32.4 ms      1.0s

Answers were identical on every device (same 32 questions, same hits; vectors matched the CPU
fp32 reference to a cosine of 1.0), so this is purely about speed and leaving the CPU alone.
Person detection stays on the NPU - it was never benchmarked elsewhere."""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = ROOT/'models/embedding/openvino/openvino_model.xml'
MAX_TOKENS = 256
PASSAGE_WORDS, PASSAGE_STEP = 160, 140
EMBEDDING_DEVICE, EMBEDDING_BATCH = 'GPU.0', 32   # Intel iGPU; batches keep it fed
PERSON_DEVICE = 'NPU'


class Embedder:
    """all-MiniLM-L6-v2: tokenize, run, mean-pool over real tokens, L2-normalize.

    The service compiles it with a fixed [batch, MAX_TOKENS] input: static shapes are what
    the iGPU and NPU want, and padding to 256 tokens does not change the vectors (a test
    checks that). `device`, `static_shape` and `batch` exist so tests and benchmarks can run
    the identical computation elsewhere; the service itself never falls back."""

    def __init__(self, core, device=EMBEDDING_DEVICE, static_shape=True, batch=EMBEDDING_BATCH,
                 cache_dir=ROOT/'compiled-cache'):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/embedding', local_files_only=True)
        self.static_shape, self.batch = static_shape, batch
        model = core.read_model(EMBEDDING_MODEL)
        dims = [batch, MAX_TOKENS] if static_shape else [-1, -1]
        model.reshape({port.get_any_name(): dims for port in model.inputs})
        config = {'CACHE_DIR': str(cache_dir)} if cache_dir else {}
        self.model = core.compile_model(model, device, config)
        self.inputs = [p.get_any_name() for p in self.model.inputs]

    def embed_many(self, texts):
        """One normalized vector per text, `batch` texts per inference call."""
        vectors = []
        for start in range(0, len(texts), self.batch):
            group = list(texts[start:start + self.batch])
            real = len(group)
            if self.static_shape:
                group += [''] * (self.batch - real)  # fixed batch: pad, then discard
                encoded = self.tokenizer(group, padding='max_length', truncation=True,
                                         max_length=MAX_TOKENS, return_tensors='np')
            else:
                encoded = self.tokenizer(group, padding=True, truncation=True,
                                         max_length=MAX_TOKENS, return_tensors='np')
            out = np.asarray(self.model({name: encoded[name] for name in self.inputs})[self.model.output(0)])
            mask = encoded['attention_mask'][..., None]
            pooled = (out * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1) if out.ndim == 3 else out
            pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.extend(pooled[:real].astype(np.float32))
        return vectors

    def __call__(self, text):
        return self.embed_many([text])[0]


def passages(text):
    """Long rows are scored as overlapping word windows; the row takes its best window."""
    words = str(text).split()
    return [' '.join(words[i:i + PASSAGE_WORDS]) for i in range(0, len(words), PASSAGE_STEP)]


def rank(query_vector, rows, embed, limit=30):
    ranked = []
    for row in rows:
        score = max((float(embed(t) @ query_vector) for t in passages(row['text'])), default=-1)
        ranked.append(dict(row, semantic_score=round(score, 4)))
    return sorted(ranked, key=lambda r: r['semantic_score'], reverse=True)[:limit]


class Runtime:
    def __init__(self):
        self.core = ov.Core()
        missing = [d for d in (EMBEDDING_DEVICE, PERSON_DEVICE) if d not in self.core.available_devices]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} unavailable; no fallback device is configured.")
        self.models = {}
        self.lock = threading.RLock()
        self.embedder = None
        self.cache = sqlite3.connect(ROOT/'embeddings.sqlite3', check_same_thread=False)
        self.cache.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value TEXT)')

    def model(self, name):
        if name not in self.models:
            files = {'person':'person.xml'}
            model = self.core.read_model(ROOT/'models'/files[name])
            self.models[name] = self.core.compile_model(model, PERSON_DEVICE, {'CACHE_DIR':str(ROOT/'compiled-cache')})
        return self.models[name]

    def embedding(self, text):
        return self.embeddings([text])[0]

    def embeddings(self, texts):
        """One vector per text, in order. Everything already embedded comes from the on-disk
        cache; the rest go to the device together, since a batch is what makes the iGPU quick
        (35s vs 45s one at a time over a whole library)."""
        # Revision is part of the cache key; replacing the model invalidates vectors.
        revision = (ROOT/'models/sources.json').read_text()
        keys = {text: hashlib.sha256((revision+text).encode()).hexdigest() for text in dict.fromkeys(texts)}
        known = {}
        wanted = list(keys.values())
        for start in range(0, len(wanted), 400):  # SQLite caps how many parameters one query takes
            group = wanted[start:start+400]
            placeholders = ','.join('?' * len(group))
            known.update(self.cache.execute(
                f'SELECT key,value FROM vectors WHERE key IN ({placeholders})', group).fetchall())
        missing = [text for text, key in keys.items() if key not in known]
        if missing:
            with self.lock:
                if self.embedder is None:
                    self.embedder = Embedder(self.core)
                vectors = self.embedder.embed_many(missing)
            rows = [(keys[text], json.dumps(vector.tolist())) for text, vector in zip(missing, vectors)]
            self.cache.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?)', rows)
            self.cache.commit()
            known.update(rows)
        return [np.asarray(json.loads(known[keys[text]]), dtype=np.float32) for text in texts]

    def run(self, action, body):
        # Let person detection interleave between search embeddings, not wait for
        # an entire library's first indexing run. Each individual inference is locked.
        from contextlib import nullcontext
        with (nullcontext() if action=='search' else self.lock):
            if action == 'search':
                query = str(body['query'])
                rows = body['rows']
                if not query.strip() or len(query)>1000 or len(rows)>10000:
                    raise ValueError('Invalid search size')
                # Embed the query and every passage in one pass, so the device gets full
                # batches instead of one short line at a time.
                texts = [query] + [t for row in rows for t in passages(row['text'])]
                vectors = dict(zip(texts, self.embeddings(texts)))
                results = rank(vectors[query], rows, vectors.__getitem__)
                return {'device':EMBEDDING_DEVICE,'results':results}
            if action == 'person':
                gray = np.asarray(body.get('bgr',body.get('gray')),dtype=np.uint8)
                if gray.ndim not in (2,3) or gray.size>640*480*3 or (gray.ndim==3 and gray.shape[2]!=3):
                    raise ValueError('Expected a small grayscale frame')
                model = self.model('person')
                _,channels,height,width = model.input(0).shape
                resized = gray[np.linspace(0,gray.shape[0]-1,height).astype(int)[:,None],
                               np.linspace(0,gray.shape[1]-1,width).astype(int)]
                batch = (np.repeat(resized[None,None],channels,axis=1) if resized.ndim==2
                         else resized.transpose(2,0,1)[None]).astype(np.float32)
                detections = np.asarray(model([batch])[model.output(0)]).reshape(-1,7)
                boxes = [r[2:].tolist() for r in detections if r[0]>=0 and r[2]>=0.6]
                return {'device':PERSON_DEVICE,'boxes':boxes,'experimental_ir':True}
            raise ValueError('Unknown action')
