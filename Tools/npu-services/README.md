# Local on-device inference (iGPU + NPU)

Shared loopback service on port 8788. Each model compiles to one named device and stays
there; AUTO, heterogeneous execution and fallback are not configured, so a missing device is
an error. Text embedding runs on the **Intel iGPU** in batches of 32; object detection runs on
the **NPU**. CPU still handles ordinary tokenization, resizing, transport, and vector
comparison. Energy use has not been measured.

Devices were chosen by measurement (lecture-notes `tools/semantic_search_benchmark.py`,
2026-09-18, 6155 passages of real notes and transcripts). Indexing the lot: Intel iGPU in
batches 35.2s wall / 11.1s CPU; NPU one at a time 55.6s / 13.8s; CPU one at a time 199.5s /
1022.3s. Answers were identical on every device, so the choice is about speed and keeping the
CPU free. Object detection (YOLOX-tiny) runs on the NPU at about 12 ms a frame; it was chosen
on 2026-09-28 over person-detection-0202 and the previous person-detection-retail-0013 by
measurement on real IR-viewer frames (see the table in `runtime.py`).

Clients start the service on demand, hidden, through `start.ps1`. Nothing starts at
Windows login. Models load lazily and inference stops when there are no requests.
The process uses the existing Lecture Notes Python environment. To reinstall,
install requirements there and run `setup_models.py`; downloaded files are pinned
by URL/revision and SHA-256 in `model-manifest.json`.

- **Notes:** choose **Meaning search** and press Enter. Keyword search remains
  available. New or edited passages are embedded once and cached. Results retain
  class/date and transcript recording links. The first search can take longer.
- **Research Library:** choose **Meaning search** under Answer style and submit
  the question. Results point to source PDF pages; they are similarity matches, not
  generated answers. The document filter is respected.
- **IR viewer:** V cycles IR / visible / both, F cycles illuminated /
  unilluminated / all IR exposures, and D cycles detection off / IR / visible / both.
  These selectors are clickable. Detection runs about 3.5 times a second per selected
  camera (a request every 250 ms, about 25 ms per round trip). The detector recognizes
  the 80 COCO object types (people, chairs, cups, phones...) through the `objects` action; the older `person` action returns people only, same format.
  IR accuracy is experimental; ordinary RGB training does not validate IR performance.

Model provenance:

- sentence-transformers/all-MiniLM-L6-v2 (OpenVINO weights), Apache-2.0.
- Megvii YOLOX-tiny (release 0.1.1rc0 ONNX), Apache-2.0.
- Intel Open Model Zoo person-detection-retail-0013, Apache-2.0 (no longer used; still in the
  manifest because `models/sources.json`, the embedding cache key, lists it).

Search quality is covered by lecture-notes `tests/test_semantic_search.py` (32 paraphrased
study questions against a frozen copy of real notes). NPU compilation and synthetic inference
were checked on Intel AI Boost; they do not establish real-world infrared detection accuracy.
All data processing is local. No photos or passages are uploaded by this service.
Failures report unavailability rather than switching model execution devices.
