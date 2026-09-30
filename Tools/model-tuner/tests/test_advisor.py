from modeltuner import advisor


def _fake_ask(choice, prob, conf):
    def ask(state, questions, timeout_s=25.0):
        assert "`facts`" in questions["cause"]["instructions"] and "`log_lines`" in questions["cause"]["instructions"]
        assert set(questions["cause"]["criteria"]) == set(advisor.FAILURE_CAUSES)
        return {"answers": {"cause": {"choice": choice, "confidence": conf,
                                      "probabilities": {choice: prob, "unknown": 1 - prob}}}}
    return ask


def test_confident_answer_is_actionable_and_mapped(monkeypatch):
    monkeypatch.setattr(advisor, "ask", _fake_ask("out_of_vram", 0.98, 0.97))
    d = advisor.diagnose_failure("E cudaMalloc failed: out of memory", {"ctx": 1})
    assert d.kind == "oom" and d.actionable and "confidence 0.97" in d.describe()


def test_diffuse_answer_is_logged_but_not_acted_on(monkeypatch):
    # the live eval's one miss: an empty log, Jev guessed out_of_vram at p=0.33, confidence 0.20
    monkeypatch.setattr(advisor, "ask", _fake_ask("out_of_vram", 0.33, 0.20))
    d = advisor.diagnose_failure("I srv load_model: loading model", {"ctx": 1})
    assert d.cause == "out_of_vram" and not d.actionable


def test_errors_and_unknown_labels_give_none(monkeypatch):
    monkeypatch.setattr(advisor, "ask", lambda *a, **k: {"error": "no key"})
    assert advisor.diagnose_failure("x", {}) is None
    monkeypatch.setattr(advisor, "ask", _fake_ask("gremlins", 0.9, 0.9))
    assert advisor.diagnose_failure("x", {}) is None


def test_filter_log_keeps_errors_and_strips_paths():
    tail = ("I srv loading model 'C:\\AI\\models\\secret-name.gguf'\n"
            "E llama_model_load: error loading model from C:\\Users\\someone\\x.gguf: failed\n")
    lines = advisor.filter_log(tail)
    assert len(lines) == 1 and "x.gguf" in lines[0] and "Users" not in lines[0]
