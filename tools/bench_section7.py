"""Measurements and simulations for proposal Section 7.5 (all on a CPU; no trained weights or datasets needed).

python tools/bench_section7.py recognizer OUT_DIR      size, compute and CPU time of the v5 recognizer architecture,
                                                       ONNX export checked against PyTorch (random weights: the
                                                       cost does not depend on the weights)
python tools/bench_section7.py goal2 MODEL_DIR OUT_DIR NLLB-200 (32-bit and CTranslate2 8-bit), Piper EN/TR/ES and a
                                                       Piper -> Whisper loopback on ten service-counter sentences
python tools/bench_section7.py summary OUT_DIR         combine both with the v5 results (docs/GOAL1_LOG.md,
                                                       docs/figures/calib_v5_report.md), run the correction simulation
                                                       and write docs/figures/section7_benchmarks.json

Needs: torch (CPU), onnx, onnxruntime; for goal2 also transformers<5, sentencepiece, ctranslate2, piper-tts,
faster-whisper, jiwer. MODEL_DIR receives about 5 GB of downloaded models.
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREADS = 4

SENTENCES = ["Hello, I would like a coffee, please.", "How much is this?", "Where is the restroom?",
             "I need help.", "Can you write it down, please?", "I want to pay with a credit card.",
             "Thank you, have a good day.", "I am waiting for my appointment.", "Can I return this?",
             "Where is the pharmacy?"]
VOICES = {"en": "en_US-lessac-medium", "tr": "tr_TR-dfki-medium", "es": "es_ES-davefx-medium"}

# v5 results on the 11 ASL Citizen test signers (docs/GOAL1_LOG.md, docs/figures/calib_v5_report.md)
V5 = dict(v4_top1=67.2, v4_top5=83.9, v5_top1=86.6, v5_top5=98.2, scratch_top1=85.7, scratch_top5=97.8)
CALIB = dict(T=0.55, ece_raw=14.6, ece_scaled=1.5, threshold=0.62, coverage=85.1, error=6.5, worst_error=12.9)
# per test signer at the 0.62 threshold: (share of signs accepted, error among accepted), in %
SIGNERS = {5: (79.9, 12.9), 8: (85.7, 7.4), 13: (86.3, 5.7), 15: (82.2, 6.7), 16: (87.9, 5.9), 19: (81.3, 6.7),
           32: (87.6, 4.3), 39: (83.6, 3.5), 44: (86.5, 4.7), 45: (88.8, 7.9), 46: (85.7, 6.2)}
# memory allowances that are assumed, not measured (GB)
MEM_ASSUMED = {"Operating system and desktop": 2.0, "GPU runtime (CUDA, TensorRT)": 0.5,
               "Landmark extractor (MediaPipe)": 0.1, "Sentence-builder language model": 0.5}


def best_of(fn, reps=3):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def cpu_name():
    out = subprocess.run("lscpu", capture_output=True, text=True).stdout
    m = re.search(r"Model name:\s*(.+)", out)
    return m.group(1).strip() if m else "unknown"


# ---------------------------------------------------------------- recognizer
def bench_recognizer(out_dir):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch
    import onnxruntime as ort
    from sign_model import SignModelV3, features_v4
    torch.manual_seed(0)
    P, MAX_LEN, N_CLASSES = 94, 64, 134
    model = SignModelV3(P * 6 + 84 + 2, N_CLASSES, 192).eval()
    x = torch.randn(1, MAX_LEN, P, 2) * 0.5
    mask = torch.ones(1, MAX_LEN, dtype=torch.bool)
    feats = features_v4(x)
    res = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    res["fp32_MB"] = res["params_M"] * 4
    onnx_path = os.path.join(out_dir, "sign_v5_arch.onnx")
    torch.onnx.export(model, (feats, mask), onnx_path, input_names=["features", "mask"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(feats, mask).numpy()
    inputs = {"features": feats.numpy(), "mask": mask.numpy()}
    for threads in (1, THREADS):
        torch.set_num_threads(threads)
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        got = sess.run(None, inputs)[0]
        res["max_abs_diff"] = float(np.abs(ref - got).max())
        ts = []
        for _ in range(320):
            t0 = time.perf_counter(); sess.run(None, inputs); ts.append(time.perf_counter() - t0)
        res[f"ort_ms_{threads}t"] = float(np.median(ts[20:]) * 1000)
        with torch.no_grad():
            tt = []
            for _ in range(120):
                t0 = time.perf_counter(); model(features_v4(x), mask); tt.append(time.perf_counter() - t0)
        res[f"torch_ms_{threads}t"] = float(np.median(tt[20:]) * 1000)
    # multiply-accumulates of the linear and convolution layers, plus the attention products
    macs = [0]

    def hook(mod, inp, out):
        if isinstance(mod, torch.nn.Linear):
            macs[0] += inp[0].numel() // inp[0].shape[-1] * mod.in_features * mod.out_features
        elif isinstance(mod, torch.nn.Conv1d):
            macs[0] += out.numel() * mod.in_channels // mod.groups * mod.kernel_size[0]
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d))]
    with torch.no_grad():
        model(feats, mask)
    for h in hooks:
        h.remove()
    n_attn = sum(isinstance(m, torch.nn.MultiheadAttention) for m in model.modules())
    macs[0] += n_attn * (2 * MAX_LEN * MAX_LEN * 192 + 4 * MAX_LEN * 192 * 192)  # QK^T, AV and projections
    res["macs_M"] = macs[0] / 1e6
    res["onnx_MB"] = os.path.getsize(onnx_path) / 1e6
    json.dump(res, open(os.path.join(out_dir, "recognizer.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- language and speech
def bench_goal2(model_dir, out_dir):
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    import torch
    torch.set_num_threads(THREADS)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    res = {"threads": THREADS, "cpu": cpu_name()}
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn", cache_dir=model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=model_dir).eval()
    res["nllb_params_M"] = sum(p.numel() for p in mdl.parameters()) / 1e6
    trans = {"tur_Latn": [], "spa_Latn": []}
    for lang in trans:
        bos = tok.convert_tokens_to_ids(lang)
        for s in SENTENCES:
            enc = tok(s, return_tensors="pt")

            def run():
                with torch.no_grad():
                    return mdl.generate(**enc, forced_bos_token_id=bos, num_beams=4, max_new_tokens=48)
            t = best_of(run)
            trans[lang].append({"en": s, "out": tok.batch_decode(run(), skip_special_tokens=True)[0], "fp32_s": t})
    del mdl
    ct2_dir = os.path.join(model_dir, "nllb600m-ct2-int8")
    if not os.path.exists(ct2_dir):
        subprocess.run(["ct2-transformers-converter", "--model", name, "--output_dir", ct2_dir, "--quantization",
                        "int8"], check=True, env={**os.environ, "HF_HOME": model_dir, "HF_HUB_CACHE": model_dir})
    import ctranslate2
    tr = ctranslate2.Translator(ct2_dir, device="cpu", intra_threads=THREADS, compute_type="int8")
    res["ct2_dir_MB"] = sum(os.path.getsize(os.path.join(ct2_dir, f)) for f in os.listdir(ct2_dir)) / 1e6
    for lang in trans:
        for row in trans[lang]:
            src = tok.convert_ids_to_tokens(tok.encode(row["en"]))
            run = lambda: tr.translate_batch([src], target_prefix=[[lang]], beam_size=4, max_decoding_length=48)
            row["int8_s"] = best_of(run)
            hyp = run()[0].hypotheses[0]
            row["int8_out"] = tok.decode(tok.convert_tokens_to_ids(hyp[1:] if hyp and hyp[0] == lang else hyp),
                                         skip_special_tokens=True)
    res["translations"] = trans

    from piper import PiperVoice
    vd = os.path.join(model_dir, "voices")
    subprocess.run([sys.executable, "-m", "piper.download_voices", "--download-dir", vd] + list(VOICES.values()),
                   check=True)
    tts, wavs = {}, {}
    for lang, v in VOICES.items():
        voice = PiperVoice.load(os.path.join(vd, v + ".onnx"))
        texts = SENTENCES if lang == "en" else [r["int8_out"] for r in trans["tur_Latn" if lang == "tr" else "spa_Latn"]]
        rows = []
        for i, text in enumerate(texts):
            t_all = best_of(lambda: list(voice.synthesize(text)))
            t_first = best_of(lambda: next(iter(voice.synthesize(text))))
            chunks = list(voice.synthesize(text))
            audio = np.concatenate([c.audio_int16_array for c in chunks]); sr = chunks[0].sample_rate
            rows.append({"text": text, "synth_s": t_all, "first_audio_s": t_first, "audio_s": len(audio) / sr,
                         "rtf": t_all / (len(audio) / sr)})
            wavs[(lang, i)] = (audio, sr)
        tts[lang] = {"voice": v, "onnx_MB": os.path.getsize(os.path.join(vd, v + ".onnx")) / 1e6, "rows": rows}
    res["tts"] = tts

    from faster_whisper import WhisperModel
    import jiwer

    def words(s):
        return re.sub(r"[^\w\s']", "", s.lower()).split()
    asr = {}
    for size in ["base", "small"]:
        wm = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=THREADS,
                          download_root=os.path.join(model_dir, "whisper"))
        out = {}
        for lang in VOICES:
            rows = []
            for i in range(len(SENTENCES)):
                audio, sr = wavs[(lang, i)]
                a = audio.astype(np.float32) / 32768
                if sr != 16000:
                    idx = np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr))
                    a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)

                def native():
                    segs, _ = wm.transcribe(a, language=lang, beam_size=5, vad_filter=False)
                    return " ".join(s.text for s in segs)

                def to_en():
                    segs, _ = wm.transcribe(a, language=lang, task="translate", beam_size=5, vad_filter=False)
                    return " ".join(s.text for s in segs)
                row = {"ref": tts[lang]["rows"][i]["text"], "s": best_of(native, 2), "hyp": native().strip(),
                       "audio_s": len(a) / 16000}
                if lang != "en":
                    row["en_s"] = best_of(to_en, 2); row["en_hyp"] = to_en().strip(); row["en_ref"] = SENTENCES[i]
                rows.append(row)
            out[lang] = {"wer": jiwer.wer([" ".join(words(r["ref"])) for r in rows],
                                          [" ".join(words(r["hyp"])) for r in rows]), "rows": rows}
        asr[size] = out
    res["asr"] = asr
    json.dump(res, open(os.path.join(out_dir, "goal2.json"), "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- summary and simulation
def dir_size_gb(path):
    """Real files only: the Hugging Face cache links each snapshot file to a blob."""
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs
               if not os.path.islink(os.path.join(dp, f))) / 1e9


def summary(out_dir, model_dir=None):
    rec = json.load(open(os.path.join(out_dir, "recognizer.json")))
    g2 = json.load(open(os.path.join(out_dir, "goal2.json")))
    med = lambda xs: float(np.median(xs))
    T = g2["translations"]
    nllb = {"tr_fp32_s": med([r["fp32_s"] for r in T["tur_Latn"]]), "tr_int8_s": med([r["int8_s"] for r in T["tur_Latn"]]),
            "es_fp32_s": med([r["fp32_s"] for r in T["spa_Latn"]]), "es_int8_s": med([r["int8_s"] for r in T["spa_Latn"]]),
            "fp32_GB": g2["nllb_params_M"] * 4 / 1000, "int8_GB": g2["ct2_dir_MB"] / 1000,
            "samples": {k: [(r["en"], r["int8_out"]) for r in v] for k, v in T.items()}}
    tts = g2["tts"]
    piper = {lang: {"first_s": med([r["first_audio_s"] for r in tts[lang]["rows"]]),
                    "synth_s": med([r["synth_s"] for r in tts[lang]["rows"]]),
                    "rtf": med([r["rtf"] for r in tts[lang]["rows"]]), "MB": tts[lang]["onnx_MB"]} for lang in tts}
    piper["first_max_s"] = max(r["first_audio_s"] for lang in tts for r in tts[lang]["rows"])
    asr = g2["asr"]
    whisper = {}
    for size in asr:
        for lang in asr[size]:
            whisper[f"{size}_wer_{lang}"] = 100 * asr[size][lang]["wer"]
            whisper[f"{size}_s_{lang}"] = med([r["s"] for r in asr[size][lang]["rows"]])
            if lang != "en":
                whisper[f"{size}_en_s_{lang}"] = med([r["en_s"] for r in asr[size][lang]["rows"]])
        whisper[f"{size}_audio_s"] = med([r["audio_s"] for lang in asr[size] for r in asr[size][lang]["rows"]])
    if model_dir:
        for size in asr:
            d = [os.path.join(model_dir, "whisper", x) for x in os.listdir(os.path.join(model_dir, "whisper")) if
                 x.endswith(f"whisper-{size}")]
            whisper[f"{size}_GB"] = dir_size_gb(d[0]) if d else None

    # latency from confirmation to first audio: translation (8-bit) + Piper's first chunk, per sentence
    lat = {}
    for lang, key in (("tr", "tur_Latn"), ("es", "spa_Latn")):
        lat[lang] = [t["int8_s"] + p["first_audio_s"] for t, p in zip(T[key], tts[lang]["rows"])]
    lat["en"] = [p["first_audio_s"] for p in tts["en"]["rows"]]

    # correction simulation: per sign, a correction is needed when the sign is not accepted (pick from the list)
    # or is accepted but wrong (replaced on the confirmation screen); signs independent
    ns = list(range(1, 7))
    per = {s: [n * ((1 - c / 100) + (c / 100) * (e / 100)) for n in ns] for s, (c, e) in SIGNERS.items()}
    none = {s: [((c / 100) * (1 - e / 100)) ** n for n in ns] for s, (c, e) in SIGNERS.items()}
    worst = max(SIGNERS, key=lambda s: per[s][2])
    avg = [float(np.mean([per[s][i] for s in SIGNERS])) for i in range(len(ns))]
    # Monte Carlo check of the closed form (100,000 three-sign sentences per signer)
    rng = np.random.default_rng(0)
    mc = []
    for s, (c, e) in SIGNERS.items():
        acc = rng.random((100_000, 3)) < c / 100
        wrong = rng.random((100_000, 3)) < e / 100
        mc.append(((~acc) | (acc & wrong)).sum(1).mean())
    mem = dict(MEM_ASSUMED)
    mem["Sign recognizer (v5)"] = rec["fp32_MB"] / 1000
    mem["NLLB-200, 8-bit"] = nllb["int8_GB"]
    mem["Whisper small"] = whisper.get("small_GB") or 0.5
    mem["Piper voices (3)"] = sum(piper[l]["MB"] for l in VOICES) / 1000
    sim = {"n": ns, "corr_avg": avg, "corr_worst": per[worst], "worst_signer": worst,
           "none_avg": [float(np.mean([none[s][i] for s in SIGNERS])) for i in range(len(ns))],
           "none_worst": none[worst], "p3_none_avg": 100 * float(np.mean([none[s][2] for s in SIGNERS])),
           "e3_avg": avg[2], "e3_worst": per[worst][2], "e3_mc_avg": float(np.mean(mc)),
           "mem": mem, "mem_total_gb": sum(mem.values()), "mem_os_gb": MEM_ASSUMED["Operating system and desktop"],
           "lat_en_s": med(lat["en"]), "lat_tr_s": med(lat["tr"]), "lat_es_s": med(lat["es"]),
           "lat_parts": {"tr": [nllb["tr_int8_s"], piper["tr"]["first_s"]], "es": [nllb["es_int8_s"], piper["es"]["first_s"]],
                         "en": [0.0, piper["en"]["first_s"]]}}
    rec = dict(rec, **V5)
    mant, ex = f"{rec['max_abs_diff']:.0e}".split("e")
    rec["max_abs_diff_str"] = f"{mant} × 10{{sup:−{int(ex[1:])}}}"  # rendered as a superscript in the proposal
    table13 = [
        ["Sign recognizer (v5), ONNX", "Classify one sign, 1 processor core", f"{rec['fp32_MB']:.1f} MB",
         f"{rec['ort_ms_1t']:.1f} ms"],
        ["NLLB-200 distilled 600M, 32-bit", "Translate English → Turkish / Spanish", f"{nllb['fp32_GB']:.1f} GB",
         f"{nllb['tr_fp32_s']:.2f} s / {nllb['es_fp32_s']:.2f} s"],
        ["NLLB-200, 8-bit (CTranslate2)", "Translate English → Turkish / Spanish", f"{nllb['int8_GB']:.2f} GB",
         f"{nllb['tr_int8_s']:.2f} s / {nllb['es_int8_s']:.2f} s"],
        ["Piper, medium voices", "First audio, English / Turkish / Spanish",
         f"{np.mean([piper[l]['MB'] for l in VOICES]):.0f} MB per voice",
         " / ".join(f"{piper[l]['first_s']:.2f}" for l in VOICES) + " s"],
        ["Piper, medium voices", "Synthesis time ÷ audio length (EN / TR / ES)", "",
         " / ".join(f"{piper[l]['rtf']:.2f}" for l in VOICES)],
    ]
    for size in ("base", "small"):
        gb = whisper.get(f"{size}_GB")
        table13.append([f"Whisper {size} (16-bit file, run in 8-bit)", "Transcribe a Turkish / Spanish reply in English",
                        f"{gb * 1000:.0f} MB" if gb else "",
                        f"{whisper[f'{size}_en_s_tr']:.2f} s / {whisper[f'{size}_en_s_es']:.2f} s"])
    out = {"cpu": g2["cpu"], "cpu_threads": g2["threads"], "recognizer": rec, "calib": CALIB, "nllb": nllb,
           "piper": piper, "whisper": whisper, "simulation": sim, "table13": table13}
    path = os.path.join(ROOT, "docs", "figures", "section7_benchmarks.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("wrote", path)
    print(json.dumps({k: v for k, v in out.items() if k not in ("table13",)}, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "recognizer":
        bench_recognizer(sys.argv[2])
    elif cmd == "goal2":
        bench_goal2(sys.argv[2], sys.argv[3])
    elif cmd == "summary":
        summary(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
