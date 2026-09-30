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

Object detection runs YOLOX-tiny (COCO, 80 classes) on the NPU, about 12 ms per frame. Picked
by measurement (2026-09-28) on 24 real IR-viewer frames of one person at a laptop, 12 infrared
and 12 visible, counting a hit as a person box overlapping the person by IoU >= 0.5 and a
false positive as any other person box (elsewhere in the frame, or on person-free background):

                                  person hits   false positives
  YOLOX-tiny                        24/24       0 at every threshold from 0.3 to 0.7
  person-detection-0202             <=17/24     never 0 at any threshold
  person-detection-retail-0013       0/24       (the old model: trained on distant shoppers)

Person scores were 0.90-0.94 in both cameras. Furniture scored 0.2-0.7, with a spurious
"couch" (really the chair and bed) never above 0.46 - hence the 0.5 threshold."""
import base64
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
OBJECT_MODEL = ROOT/'models/yolox_tiny.onnx'
OBJECT_DEVICE = 'NPU'
PERSON_DEVICE = OBJECT_DEVICE  # older name, kept for anything that still imports it
OBJECT_THRESHOLD, OBJECT_NMS_IOU = 0.5, 0.45
MAX_FRAME = 640*480*3
COCO_LABELS = (
    'person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant '
    'stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe '
    'backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat '
    'baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl '
    'banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant '
    'bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink '
    'refrigerator book clock vase scissors teddy_bear hair_drier toothbrush').split()


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


def decode_frame(body):
    """A uint8 frame from a request: raw bytes + shape (current clients), or the older nested
    JSON lists under 'gray'/'bgr'. 2-D is grayscale; 3-D must be BGR."""
    if 'pixels' in body:
        shape = tuple(int(v) for v in body['shape'])
        data = base64.b64decode(body['pixels'], validate=True)
        if len(shape) not in (2, 3) or int(np.prod(shape)) != len(data):
            raise ValueError('pixels do not match shape')
        frame = np.frombuffer(data, np.uint8).reshape(shape)
    else:
        frame = np.asarray(body.get('bgr', body.get('gray')), dtype=np.uint8)
    if frame.ndim not in (2, 3) or frame.size > MAX_FRAME or (frame.ndim == 3 and frame.shape[2] != 3):
        raise ValueError('Expected a small grayscale or BGR frame')
    return frame


def nms(boxes, scores, iou_limit):
    """Indices of the boxes to keep, best first (greedy non-maximum suppression)."""
    order, keep = scores.argsort()[::-1], []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        overlap = w * h / np.maximum(areas[i] + areas[rest] - w * h, 1e-9)
        order = rest[overlap <= iou_limit]
    return keep


class ObjectDetector:
    """YOLOX-tiny, Megvii's released ONNX: BGR 0-255 in, letterboxed into the top-left of a
    square padded with 114; out come raw per-cell predictions, decoded here (grid offset and
    stride for the centre, exp for the size) with objectness x class score as the confidence.

    Clients should send frames already shrunk to fit the input (the IR viewer does, with an
    area resize); anything else gets a nearest-neighbour resize, as this environment has no cv2."""

    def __init__(self, core, device=OBJECT_DEVICE, cache_dir=ROOT/'compiled-cache'):
        config = {'CACHE_DIR': str(cache_dir)} if cache_dir else {}
        self.model = core.compile_model(core.read_model(OBJECT_MODEL), device, config)
        self.size = int(self.model.input(0).shape[2])
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = self.size // stride
            ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def __call__(self, frame, threshold=OBJECT_THRESHOLD):
        """[(label, score, x1, y1, x2, y2)], best first, with corners as fractions of the frame."""
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        h, w = frame.shape[:2]
        ratio = min(self.size / h, self.size / w)
        rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
        if (rh, rw) != (h, w):
            frame = frame[np.linspace(0, h - 1, rh).astype(int)[:, None], np.linspace(0, w - 1, rw).astype(int)]
        padded = np.full((self.size, self.size, 3), 114, np.uint8)
        padded[:rh, :rw] = frame
        raw = np.asarray(self.model([padded.transpose(2, 0, 1)[None].astype(np.float32)])[self.model.output(0)])[0]
        centre = (raw[:, :2] + self.grid) * self.stride
        size = np.exp(raw[:, 2:4]) * self.stride
        scores = raw[:, 4:5] * raw[:, 5:]
        classes, confidence = scores.argmax(1), scores.max(1)
        keep = confidence >= threshold
        boxes = np.concatenate([centre - size / 2, centre + size / 2], 1)[keep] / ratio
        classes, confidence = classes[keep], confidence[keep]
        found = []
        for cls in np.unique(classes):
            idx = np.where(classes == cls)[0]
            for i in nms(boxes[idx], confidence[idx], OBJECT_NMS_IOU):
                x1, y1, x2, y2 = boxes[idx[i]] / (w, h, w, h)
                found.append((COCO_LABELS[cls], float(confidence[idx[i]]),
                              *(float(np.clip(v, 0, 1)) for v in (x1, y1, x2, y2))))
        return sorted(found, key=lambda d: -d[1])


class Runtime:
    def __init__(self):
        self.core = ov.Core()
        missing = [d for d in (EMBEDDING_DEVICE, OBJECT_DEVICE) if d not in self.core.available_devices]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} unavailable; no fallback device is configured.")
        self.models = {}
        self.lock = threading.RLock()
        self.embedder = None
        self.cache = sqlite3.connect(ROOT/'embeddings.sqlite3', check_same_thread=False)
        self.cache.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value TEXT)')

    def detector(self):
        if 'objects' not in self.models:
            self.models['objects'] = ObjectDetector(self.core)
        return self.models['objects']

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
            if action in ('objects', 'person'):
                found = self.detector()(decode_frame(body))
                # 'boxes' keeps the original person-only format: [score, x1, y1, x2, y2]
                people = [[score, *box] for label, score, *box in found if label == 'person']
                reply = {'device': OBJECT_DEVICE, 'model': 'yolox-tiny', 'boxes': people, 'experimental_ir': True}
                if action == 'objects':
                    reply['objects'] = [{'label': label, 'score': round(score, 3), 'box': [round(v, 4) for v in box]}
                                        for label, score, *box in found]
                return reply
            raise ValueError('Unknown action')
