"""PathSnap live pipeline monitor (pure Python: stdlib http.server + psutil, no build step).

usage: python scripts/monitor.py [port]       then open http://localhost:8765

Every job is discovered from three things, so new work shows up without editing this file:
  * running processes   - any `scripts/NN_name.py` process (and the uvicorn API) is a live job
  * log files           - results/run_<model>.log, results/bayes_<model>.log, results/<job>_run.log
  * output files        - a job is "done" once its artifact exists (comparison csv, tuned csv, figures ...)
In-progress jobs are sorted to the top, finished jobs to the bottom.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
MODELS = [("logreg", "Logistic Regression"), ("random_forest", "Random Forest"), ("lightgbm", "LightGBM"),
          ("xgboost", "XGBoost"), ("mlp", "MLP (64-32)")]
MODEL_NAME = dict(MODELS)

FOLD = re.compile(r"fold (\d): macroF1=([\d.]+) top1\(reach\)=([\d.]+) ll=([\d.]+) cont=([\d.]+) t=(\d+)s")
TEST = re.compile(r"TEST: macroF1=([\d.]+) top1=([\d.]+) ll=([\d.]+) cont=([\d.]+)")
OPTUNA = re.compile(r"Trial (\d+) finished with value: ([\d.]+)")
EPOCH = re.compile(r"epoch (\d+): train_loss=([\d.]+) val_loss=([\d.]+)")

# static pipeline order -> used for tie-breaks and for the ribbon
ORDER = ["data", "eda"] + [f"model:{k}" for k, _ in MODELS] + ["tune:lightgbm", "tune:xgboost", "decoder", "final",
                                                                  "api", "apicheck", "maps", "figures", "docs", "git",
                                                                  "handoff:eda", "handoff:errors", "handoff:tests"]

CPU_HIST = deque(maxlen=120)


def _sampler():
    while True:
        CPU_HIST.append(psutil.cpu_percent(interval=2))


def exists(*p):
    return os.path.exists(os.path.join(ROOT, *p))


def read(path):
    try:
        return open(path, errors="ignore").read()
    except OSError:
        return ""


def count_glob(folder, ext):
    d = os.path.join(ROOT, folder)
    return len([f for f in os.listdir(d) if f.endswith(ext)]) if os.path.isdir(d) else 0


def read_csv_rows(path):
    import csv
    try:
        return list(csv.DictReader(open(path, newline="")))
    except OSError:
        return []


def scan_processes():
    """Return {job_key: info} for live pipeline processes."""
    found = {}
    for p in psutil.process_iter(["name"]):          # cheap name filter first: cmdline of every process is slow
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            cmd = p.cmdline()
            p.info["create_time"] = p.create_time()
            p.info["memory_info"] = p.memory_info()
        except psutil.Error:
            continue
        if not cmd:
            continue
        joined = " ".join(cmd)
        script = next((os.path.basename(c) for c in cmd if c.endswith(".py") and "scripts" in c.replace("\\", "/")), None)
        key = None
        args = {}
        for i, c in enumerate(cmd):
            if c.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                args[c[2:]] = cmd[i + 1]
        if "uvicorn" in joined and "api" in joined:
            key = "api"
        elif script in ("monitor.py", "tui.py"):
            continue
        elif script == "02_compare_models.py":
            key = f"model:{args.get('models', 'all').split(',')[0]}"
        elif script == "03_bayes_opt.py":
            key = f"tune:{args.get('model', 'lightgbm')}"
        elif script == "04_tune_decoder.py":
            key = "decoder"
        elif script == "01_build_dataset.py":
            key = "data"
        elif script in ("05_eda_plots.py", "00_eda_stats.py"):
            key = "eda"
        elif script in ("06_figures.py",):
            key = "figures"
        elif script and script.startswith("08_maps"):
            key = "maps"
        elif script == "07_api_check.py":
            key = "apicheck"
        elif script and script != "02b_merge_results.py":
            key = "adhoc:" + script
        if key is None:
            continue
        mem = p.info["memory_info"].rss if p.info["memory_info"] else 0
        cur = found.get(key)
        if cur is None or mem > cur["mem"]:
            found[key] = dict(pid=p.pid, start=p.info["create_time"], mem=mem, args=args, script=script)
    return found


def git_state():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    commits = run("rev-list", "--count", "HEAD")
    head = run("rev-parse", "HEAD")
    remote = run("rev-parse", "origin/main")
    dirty = len(run("status", "--porcelain").splitlines())
    return dict(commits=int(commits) if commits.isdigit() else 0, pushed=bool(head) and head == remote, dirty=dirty)


HANDOFFS = [("eda", "handoff-eda", "Teammate 1: EDA overview + findings"),
            ("errors", "handoff-errors", "Teammate 2: error analysis"),
            ("tests", "handoff-tests-runbook", "Teammate 3: tests + demo runbook")]
HANDOFF_CACHE = {}


def refresh_handoffs():
    """git fetch (read-only) then summarise each handoff branch; called from the background thread."""
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=40).stdout.strip()
        except Exception:
            return ""
    run("fetch", "origin", "--quiet", "--prune")
    for key, br, _ in HANDOFFS:
        ref = f"origin/{br}"
        if not run("rev-parse", "--verify", "--quiet", ref):
            HANDOFF_CACHE[key] = dict(exists=False)
            continue
        info = run("log", "-1", "--format=%an|%ar|%s", ref).split("|", 2)
        ahead = run("rev-list", "--count", f"origin/main..{ref}")
        files = run("diff", "--name-only", f"origin/main...{ref}").splitlines()
        merged = subprocess.run(["git", "merge-base", "--is-ancestor", ref, "origin/main"], cwd=ROOT).returncode == 0
        HANDOFF_CACHE[key] = dict(exists=True, author=info[0] if info else "", when=info[1] if len(info) > 1 else "",
                                  subject=info[2] if len(info) > 2 else "", ahead=int(ahead or 0), files=files, merged=merged)


def api_listening():
    try:
        return any(c.laddr and c.laddr.port == 8000 and c.status == "LISTEN" for c in psutil.net_connections(kind="tcp"))
    except Exception:
        return False


def job(key, name, kind, state, **kw):
    return dict(key=key, name=name, kind=kind, state=state, **kw)


def build_jobs():
    live = scan_processes()
    jobs = []
    now = time.time()

    def run_info(key):
        p = live.get(key)
        return dict(elapsed=int(now - p["start"]), pid=p["pid"]) if p else {}

    # ---- data build / EDA
    done = exists("data", "processed", "train.pkl") and exists("data", "processed", "student_test.pkl")
    jobs.append(job("data", "Candidate + feature dataset", "data", "running" if "data" in live else ("done" if done else "waiting"),
                    detail="road-arc candidates, labels and 28 features for 80 + 10 trips", **run_info("data")))
    eda_done = exists("results", "eda", "trajectories_train.png") and count_glob("results/maps", ".html") > 0
    jobs.append(job("eda", "Exploratory data analysis", "data", "running" if "eda" in live else ("done" if eda_done else "waiting"),
                    detail=f"{count_glob('results/eda', '.png')} plots, {count_glob('results/maps', '.html')} Folium maps", **run_info("eda")))

    # ---- model comparison
    for k, name in MODELS:
        log = read(os.path.join(RES, f"run_{k}.log"))
        folds = [dict(k=int(m[1]), f1=float(m[2]), top1=float(m[3]), ll=float(m[4]), cont=float(m[5]), t=int(m[6])) for m in FOLD.finditer(log)]
        t = TEST.search(log)
        test = dict(f1=float(t[1]), top1=float(t[2]), ll=float(t[3]), cont=float(t[4])) if t else None
        epochs = [(int(a), float(b), float(c)) for a, b, c in EPOCH.findall(log)]
        csvp = os.path.join(RES, "models", f"comparison_{k}.csv")
        merged = read_csv_rows(os.path.join(RES, "models", "comparison.csv"))      # consolidated table written by 02b
        done = os.path.exists(csvp) or any(r.get("model") == k for r in merged)
        failed = "Traceback" in log and f"model:{k}" not in live and not done
        state = "running" if f"model:{k}" in live else "done" if done else "failed" if failed else "waiting"
        pct = 100 if done else min(96, round(len(folds) / 6 * 100))
        ri = run_info(f"model:{k}")
        if not ri and os.path.exists(os.path.join(RES, f"run_{k}.log")) and done:
            ri = dict(elapsed=int(os.path.getmtime(os.path.join(RES, f"run_{k}.log")) - os.path.getctime(os.path.join(RES, f"run_{k}.log"))))
        jobs.append(job(f"model:{k}", name, "model", state, folds=folds, test=test, epochs=epochs, pct=pct,
                        error=log.strip().splitlines()[-1][:200] if failed else None,
                        detail="5-fold GroupKFold by trip, then refit and student-test", **ri))

    # ---- Bayesian optimisation
    tuned_models = {"xgboost"} | {f[6:-4] for f in os.listdir(RES) if f.startswith("bayes_") and f.endswith(".log")}         | {key[5:] for key in live if key.startswith("tune:")}
    for k in sorted(tuned_models):
        log = read(os.path.join(RES, f"bayes_{k}.log"))
        trials = [(int(a), float(b)) for a, b in OPTUNA.findall(log)]
        rows = read_csv_rows(os.path.join(RES, "bayes", k, "tuned_vs_untuned.csv"))
        n_planned = int((live.get(f"tune:{k}") or {}).get("args", {}).get("trials", 30))
        search_over = bool(rows) or os.path.exists(os.path.join(RES, "bayes", k, "best_params.json"))
        n_total = len(trials) if search_over else max(n_planned, len(trials))
        done = bool(rows)
        failed = "Traceback" in log and f"tune:{k}" not in live and not done
        state = "running" if f"tune:{k}" in live else "done" if done else "failed" if failed else ("waiting" if not log else "waiting")
        if log and not done and f"tune:{k}" not in live and not failed:
            state = "waiting"
        pct = 100 if done else (90 if search_over else (min(88, round(len(trials) / n_total * 88)) if trials else (3 if state == "running" else 0)))
        bp = {}
        try:
            bp = json.load(open(os.path.join(RES, "bayes", k, "best_params.json")))
        except Exception:
            pass
        tv = {r["variant"]: r for r in rows}
        jobs.append(job(f"tune:{k}", f"Optuna TPE: {MODEL_NAME.get(k, k)}", "tuning", state, trials=sorted(trials), n_total=n_total, pct=pct,
                        best=max([v for _, v in trials], default=None), best_params=bp.get("best_params"),
                        tuned=tv, error=log.strip().splitlines()[-1][:200] if failed else None,
                        detail=("search finished, comparing tuned vs untuned" if search_over and not done else "maximise 5-fold group CV macro-F1"),
                        search_over=search_over, **run_info(f"tune:{k}")))

    # ---- decoder
    log = read(os.path.join(RES, "decoder_run.log"))
    trials = [(int(a), float(b)) for a, b in OPTUNA.findall(log)]
    n_total = int((live.get("decoder") or {}).get("args", {}).get("trials", 50))
    rows = read_csv_rows(os.path.join(RES, "hmm", "summary.csv"))
    done = bool(rows)
    failed = "Traceback" in log and "decoder" not in live and not done
    st = "running" if "decoder" in live else "done" if done else "failed" if failed else "waiting"
    jobs.append(job("decoder", "HMM / Viterbi decoder tuning", "decoder", st, trials=sorted(trials), n_total=n_total,
                    pct=100 if done else (min(90, round(len(trials) / n_total * 90)) if trials else (4 if st == "running" else 0)),
                    best=max([v for _, v in trials], default=None), summary=rows, error=log.strip().splitlines()[-1][:200] if failed else None,
                    detail="out-of-fold scores, tuned on training trips only", **run_info("decoder")))

    # ---- final model package
    fin = exists("models", "final", "model.joblib") and exists("models", "final", "decoder.json")
    dec = {}
    if fin:
        try:
            dec = json.load(open(os.path.join(ROOT, "models", "final", "decoder.json")))
        except Exception:
            pass
    ready = fin and bool(dec.get("decoder_params"))
    jobs.append(job("final", "Final model package", "model", "done" if ready else "waiting",
                    detail=("model: " + dec.get("model_name", "?") + " + decoder params") if ready else "models/final/model.joblib + decoder.json"))

    # ---- API + UI
    listening = api_listening() or "api" in live
    chk = read_csv_rows(os.path.join(RES, "api_check.csv"))
    jobs.append(job("api", "FastAPI server (Laptop 1)", "service", "running" if listening else ("done" if chk else "waiting"),
                    detail="serving :8000" if listening else "start: uvicorn api.main:app --host 0.0.0.0 --port 8000", live=listening, **run_info("api")))
    jobs.append(job("apicheck", "End-to-end API check (all trips)", "check", "running" if "apicheck" in live else "done" if chk else "waiting",
                    rows=chk, detail=f"{len(chk)} trips answered" if chk else "POST /predict for every served trip", **run_info("apicheck")))

    # ---- maps / figures / docs / git
    maps = [f for f in (os.listdir(os.path.join(RES, "maps")) if os.path.isdir(os.path.join(RES, "maps")) else []) if f.startswith("trip_") or f.startswith("pred_")]
    jobs.append(job("maps", "Prediction maps (Folium)", "report", "running" if "maps" in live else "done" if maps else "waiting",
                    detail=f"{len(maps)} prediction maps", **run_info("maps")))
    figs = count_glob("results/figures", ".png")
    jobs.append(job("figures", "Comparison figures", "report", "running" if "figures" in live else "done" if figs >= 8 else "waiting",
                    detail=f"{figs} figures", **run_info("figures")))
    jobs.append(job("docs", "README + report notes", "report", "done" if exists("README.md") else "waiting", detail="README.md"))
    g = git_state()
    gstate = "done" if g["pushed"] and g["dirty"] == 0 and g["commits"] else "waiting"
    jobs.append(job("git", "Commit + push to GitHub", "release", gstate,
                    detail=(f"{g['commits']} commits, {'pushed' if g['pushed'] else 'not pushed'}, {g['dirty']} uncommitted files")))

    # ---- teammate handoff branches
    for key, br, title in HANDOFFS:
        h = HANDOFF_CACHE.get(key)
        if not h or not h.get("exists"):
            jobs.append(job(f"handoff:{key}", title, "teammate", "waiting", detail=f"no branch '{br}' on GitHub yet"))
        else:
            st = "done" if h["merged"] else "running"
            jobs.append(job(f"handoff:{key}", title, "teammate", st,
                            detail=f"{h['author']} pushed {h['when']}: {h['subject'][:60]} ({h['ahead']} commits, {len(h['files'])} files)"
                                   + (", merged" if h["merged"] else ", awaiting review")))

    # ---- ad hoc processes (anything else started under scripts/)
    for key, p in live.items():
        if key.startswith("adhoc:"):
            jobs.append(job(key, p["script"], "adhoc", "running", detail="running script", elapsed=int(now - p["start"]), pid=p["pid"]))

    rank = {k: i for i, k in enumerate(ORDER)}
    for j in jobs:
        j["rank"] = rank.get(j["key"], 99)
    return jobs


SNAPSHOT = {"jobs": [], "cpu": [], "cpu_hist": [], "cpu_avg": 0, "mem_pct": 0, "mem_gb": 0, "mem_total": 0, "time": "starting"}


def _refresher():
    """Build the status in the background so HTTP requests are answered instantly from the cache."""
    global SNAPSHOT
    last_fetch = 0.0
    while True:
        try:
            if time.time() - last_fetch > 60:
                last_fetch = time.time()
                threading.Thread(target=refresh_handoffs, daemon=True).start()
            SNAPSHOT = status()
        except Exception as e:  # keep the monitor alive whatever a job's files look like
            SNAPSHOT = dict(SNAPSHOT, time="error: " + str(e)[:60])
        time.sleep(2)


def status():
    cpu = psutil.cpu_percent(percpu=True)
    vm = psutil.virtual_memory()
    jobs = build_jobs()
    return dict(time=time.strftime("%H:%M:%S"), cpu=cpu, cpu_avg=sum(cpu) / len(cpu), cpu_hist=list(CPU_HIST),
                mem_pct=vm.percent, mem_gb=round(vm.used / 2**30, 1), mem_total=round(vm.total / 2**30), jobs=jobs)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap Pipeline Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* Layout: summary ribbon on top, jobs sorted by urgency (running first, done last), system rail on the right. */
:root{
  --bg:#edf0f3; --surface:#fafbfc; --surface-2:#f1f4f7; --ink:#15202b; --muted:#5b6a78; --line:#d3dae1;
  --accent:#1d6fb5; --accent-soft:#dcebf7;
  --run:#a85f00; --run-soft:#fbe9cc; --ok:#17775f; --ok-soft:#d7efe8; --bad:#b0372b; --bad-soft:#f8dad6; --wait:#66778a; --wait-soft:#e4e9ee;
  --font:"IBM Plex Sans",system-ui,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  --shadow:0 1px 2px rgba(21,32,43,.06),0 6px 18px -10px rgba(21,32,43,.18);
  color-scheme:light;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0c131a; --surface:#131d27; --surface-2:#1a2733; --ink:#e3eaf0; --muted:#8798a8; --line:#243342;
  --accent:#62aee8; --accent-soft:#17324a;
  --run:#f2b550; --run-soft:#3a2b0f; --ok:#45c4a3; --ok-soft:#123429; --bad:#f2796c; --bad-soft:#3d1a17; --wait:#8393a3; --wait-soft:#1f2c39;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 22px -12px rgba(0,0,0,.6); color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0c131a; --surface:#131d27; --surface-2:#1a2733; --ink:#e3eaf0; --muted:#8798a8; --line:#243342;
  --accent:#62aee8; --accent-soft:#17324a;
  --run:#f2b550; --run-soft:#3a2b0f; --ok:#45c4a3; --ok-soft:#123429; --bad:#f2796c; --bad-soft:#3d1a17; --wait:#8393a3; --wait-soft:#1f2c39;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 22px -12px rgba(0,0,0,.6); color-scheme:dark}
*{box-sizing:border-box} html,body{margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 var(--font);padding-block:18px 40px;
  padding-left:max(18px,env(safe-area-inset-left,0px));padding-right:max(18px,env(safe-area-inset-right,0px))}
.mono{font-family:var(--mono)} .num{font-variant-numeric:tabular-nums}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:19px;margin:0;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12.5px}
.grow{flex:1}
button.theme{background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font:12px var(--font);cursor:pointer}
button.theme:hover{border-color:var(--accent)} button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.ribbon{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow);margin-bottom:16px}
.ribbon-top{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;margin-bottom:9px}
.ribbon-top b{font-size:22px;font-weight:600} .ribbon-top span{color:var(--muted);font-size:12.5px}
.seg{display:flex;gap:3px;height:12px}
.seg i{flex:1;border-radius:2px;background:var(--wait-soft);transition:background .5s}
.seg i.done{background:var(--ok)} .seg i.running{background:var(--run);animation:pulse 1.6s ease-in-out infinite}
.seg i.failed{background:var(--bad)}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;align-items:start}
@media (max-width:900px){.layout{grid-template-columns:minmax(0,1fr)}}
#jobs{display:flex;flex-direction:column;gap:10px}
.sec-label{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:6px 2px 0;font-weight:600}
.job{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow);min-width:0}
.job.running{border-color:var(--run)} .job.failed{border-color:var(--bad)}
.job.done{box-shadow:none;background:var(--surface-2);padding:9px 14px}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.job h2{font-size:14.5px;margin:0;font-weight:600} .kind{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.pill{font-size:11px;font-weight:600;padding:2px 9px;border-radius:99px;display:inline-flex;gap:6px;align-items:center}
.pill.running{background:var(--run-soft);color:var(--run)} .pill.done{background:var(--ok-soft);color:var(--ok)}
.pill.failed{background:var(--bad-soft);color:var(--bad)} .pill.waiting{background:var(--wait-soft);color:var(--wait)}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor} .running .dot{animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
.meta{color:var(--muted);font-size:12px;margin-top:3px}
.bar{height:6px;background:var(--wait-soft);border-radius:3px;overflow:hidden;margin:9px 0 4px}
.bar i{display:block;height:100%;background:var(--run);border-radius:3px;transition:width .7s}
.done .bar i{background:var(--ok)}
.chips{display:flex;gap:8px 14px;flex-wrap:wrap;margin-top:8px}
.chip{font-size:12px;color:var(--muted)} .chip b{color:var(--ink);font-weight:500;font-family:var(--mono);font-size:12px}
.folds{display:flex;gap:6px;align-items:flex-end;height:46px;margin-top:10px}
.fold{flex:1;max-width:46px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px;height:100%}
.fold i{display:block;width:100%;background:var(--accent);border-radius:3px 3px 0 0;min-height:3px}
.fold span{font:10px var(--mono);color:var(--muted)}
.fold.empty i{background:var(--wait-soft)}
svg.spark{width:100%;height:auto;display:block;margin-top:8px}
.err{margin-top:8px;color:var(--bad);font:12px var(--mono);overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}
td,th{padding:3px 6px;text-align:right;border-bottom:1px solid var(--line)} th{color:var(--muted);font-weight:500}
th:first-child,td:first-child{text-align:left} td{font-family:var(--mono)} td:first-child{font-family:var(--font)}
.rail{display:flex;flex-direction:column;gap:12px;position:sticky;top:14px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow)}
.panel h3{margin:0 0 8px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600}
.big{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.cores{display:grid;grid-template-columns:repeat(11,1fr);gap:3px;margin-top:8px}
.core{height:34px;background:var(--wait-soft);border-radius:2px;position:relative;overflow:hidden}
.core i{position:absolute;left:0;right:0;bottom:0;background:var(--accent);transition:height .6s}
.proc{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid var(--line)} .proc:last-child{border:0}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<header>
  <h1>PathSnap pipeline</h1><span class="sub" id="clock"></span><span class="sub">auto-refresh 5 s</span>
  <span class="grow"></span><button class="theme" id="theme" aria-label="Toggle colour theme">Theme</button>
</header>
<section class="ribbon" aria-label="Overall progress">
  <div class="ribbon-top"><div><b id="done-n">0</b><span> / <span id="total-n">0</span> stages complete</span></div>
    <div><b id="run-n" style="color:var(--run)">0</b><span> running</span></div>
    <div><b id="fail-n" style="color:var(--bad)">0</b><span> failed</span></div></div>
  <div class="seg" id="seg"></div>
</section>
<div class="layout">
  <main id="jobs"></main>
  <aside class="rail">
    <div class="panel"><h3>CPU</h3><div><span class="big" id="cpuavg">-</span> <span class="sub">avg of <span id="ncpu">0</span> logical cores</span></div>
      <svg class="spark" id="cpuhist" viewBox="0 0 240 64" preserveAspectRatio="none"></svg><div class="cores" id="cores"></div></div>
    <div class="panel"><h3>Memory</h3><div><span class="big" id="mem">-</span> <span class="sub" id="memtxt"></span></div>
      <div class="bar"><i id="membar" style="background:var(--accent)"></i></div></div>
    <div class="panel"><h3>Live processes</h3><div id="procs" class="sub">none</div></div>
  </aside>
</div>
<script>
const $=id=>document.getElementById(id);
const f=(x,d=4)=>x==null||isNaN(x)?'-':Number(x).toFixed(d);
const dur=s=>s==null?'':s<60?s+'s':s<3600?Math.floor(s/60)+'m '+(s%60)+'s':Math.floor(s/3600)+'h '+Math.floor(s%3600/60)+'m';
const KIND={data:'Data',model:'Model',tuning:'Bayesian search',decoder:'Decoder',service:'Service',teammate:'Teammate',check:'Check',report:'Report',release:'Release',adhoc:'Script'};
const GROUP={running:0,failed:1,waiting:2,done:3}, GLABEL={0:'In progress',1:'Needs attention',2:'Queued',3:'Completed'};
// theme toggle (remembered when storage is available)
(function(){let t=null;try{t=localStorage.getItem('ps-theme')}catch(e){} if(t)document.documentElement.dataset.theme=t;
 $('theme').onclick=()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('ps-theme',nx)}catch(e){}}})();

function spark(trials,total,best){ // convergence: points + best-so-far, scaled to the data
  if(!trials||!trials.length)return '';const W=600,H=90,P=8;const ys=trials.map(t=>t[1]);let lo=Math.min(...ys),hi=Math.max(...ys);if(hi-lo<1e-6){hi=lo+1e-3}
  const X=i=>P+i*(W-2*P)/Math.max(total-1,1),Y=v=>H-P-(v-lo)/(hi-lo)*(H-2*P);let m=-1,path='',dots='';
  trials.forEach((t,i)=>{m=Math.max(m,t[1]);path+=(i?'L':'M')+X(i).toFixed(1)+' '+Y(m).toFixed(1);dots+=`<circle cx="${X(i).toFixed(1)}" cy="${Y(t[1]).toFixed(1)}" r="3.2" fill="var(--accent)" opacity=".6"/>`});
  const last=trials.length-1;
  return `<svg class="spark" viewBox="0 0 ${W} ${H}"><line x1="0" x2="${W}" y1="${H-P}" y2="${H-P}" stroke="var(--line)"/>${dots}<path d="${path}" fill="none" stroke="var(--ok)" stroke-width="2.4"/><circle cx="${X(last).toFixed(1)}" cy="${Y(m).toFixed(1)}" r="4.6" fill="var(--ok)"/></svg>`;}
function chips(o){return '<div class="chips">'+o.map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join('')+'</div>'}

function body(j){
  let h='';
  const meta=[j.detail, j.elapsed!=null?(j.state==='running'?'running ':'took ')+dur(j.elapsed):null, j.pid&&j.state==='running'?'pid '+j.pid:null].filter(Boolean).join(' · ');
  h+=`<div class="meta">${meta}</div>`;
  if(j.state==='running'||j.state==='waiting'&&j.pct)h+=`<div class="bar"><i style="width:${j.pct||4}%"></i></div>`;
  if(j.kind==='model'&&j.folds){
    if(j.state!=='done'||true){const bars=[0,1,2,3,4].map(i=>{const x=j.folds.find(q=>q.k===i);return x?`<div class="fold"><i style="height:${Math.max(8,(x.f1-.8)/.15*100)}%" title="fold ${i}"></i><span>${f(x.f1,3)}</span></div>`:`<div class="fold empty"><i style="height:8%"></i><span>fold ${i}</span></div>`}).join('');h+=`<div class="folds">${bars}</div>`}
    if(j.test)h+=chips([['test macro-F1',f(j.test.f1)],['test top-1',f(j.test.top1)],['log loss',f(j.test.ll)],['continuity',f(j.test.cont,3)]]);
    if(j.epochs&&j.epochs.length)h+=`<div class="meta">epoch ${j.epochs[j.epochs.length-1][0]} · train ${f(j.epochs[j.epochs.length-1][1])} · val ${f(j.epochs[j.epochs.length-1][2])}</div>`;
  }
  if(j.kind==='tuning'){
    h+=spark(j.trials,j.n_total,j.best);
    if(j.trials&&j.trials.length)h+=chips([['trials',j.trials.length+'/'+j.n_total],['best CV macro-F1',f(j.best,5)]]);
    if(j.tuned&&j.tuned.tuned){const u=j.tuned.untuned,t=j.tuned.tuned;
      h+=`<table><tr><th></th><th>CV macro-F1</th><th>test top-1</th><th>test macro-F1</th><th>log loss</th></tr><tr><td>untuned</td><td>${f(u.cv_macro_f1)}</td><td>${f(u.test_top1_all)}</td><td>${f(u.test_macro_f1)}</td><td>${f(u.test_log_loss)}</td></tr><tr><td><b>tuned</b></td><td>${f(t.cv_macro_f1)}</td><td>${f(t.test_top1_all)}</td><td>${f(t.test_macro_f1)}</td><td>${f(t.test_log_loss)}</td></tr></table>`}
  }
  if(j.kind==='decoder'){
    h+=spark(j.trials,j.n_total,j.best);
    if(j.trials&&j.trials.length)h+=chips([['trials',j.trials.length+'/'+j.n_total],['best accuracy',f(j.best,5)]]);
    const s=(j.summary||[]).filter(r=>r.partition==='student_test');
    if(s.length)h+=`<table><tr><th>student-test</th><th>point acc</th><th>continuity</th><th>route F1</th></tr>${s.map(r=>`<tr><td>${r.method.replace(/_/g,' ')}</td><td>${f(r.point_acc)}</td><td>${f(r.continuity)}</td><td>${f(r.route_f1)}</td></tr>`).join('')}</table>`;
  }
  if(j.kind==='check'&&j.rows&&j.rows.length){const a=k=>j.rows.reduce((s,r)=>s+parseFloat(r[k]),0)/j.rows.length;
    h+=chips([['trips',j.rows.length],['mean classifier acc',f(a('clf_acc'))],['mean +HMM acc',f(a('hmm_acc'))],['mean server time',f(a('server_s'),2)+'s']]);}
  if(j.error)h+=`<div class="err">${j.error}</div>`;
  return h;
}
function head(j){return `<div class="row"><h2>${j.name}</h2><span class="kind">${KIND[j.kind]||j.kind}</span><span style="flex:1"></span><span class="pill ${j.state}"><span class="dot"></span>${j.state}</span></div>`}

const cache={};
function render(s){
  $('clock').textContent=s.time;
  const jobs=s.jobs,n=jobs.length,c=k=>jobs.filter(j=>j.state===k).length;
  $('done-n').textContent=c('done');$('total-n').textContent=n;$('run-n').textContent=c('running');$('fail-n').textContent=c('failed');
  $('seg').innerHTML=[...jobs].sort((a,b)=>a.rank-b.rank).map(j=>`<i class="${j.state}" title="${j.name}: ${j.state}"></i>`).join('');
  const box=$('jobs');jobs.sort((a,b)=>GROUP[a.state]-GROUP[b.state]||a.rank-b.rank);
  const seen=new Set();let lastG=-1,idx=0;
  const labels=box.querySelectorAll('.sec-label');labels.forEach(l=>l.remove());
  jobs.forEach(j=>{const g=GROUP[j.state];let el=document.getElementById('job-'+j.key);
    if(!el){el=document.createElement('section');el.id='job-'+j.key;}
    const html=head(j)+body(j),sig=j.state+html;
    if(cache[j.key]!==sig){el.innerHTML=html;el.className='job '+j.state;cache[j.key]=sig;}
    if(g!==lastG){const l=document.createElement('div');l.className='sec-label';l.textContent=GLABEL[g];box.insertBefore(l,box.children[idx]||null);idx++;lastG=g;}
    if(box.children[idx]!==el)box.insertBefore(el,box.children[idx]||null);idx++;seen.add(el.id);});
  [...box.querySelectorAll('.job')].forEach(e=>{if(!seen.has(e.id))e.remove()});
  // system rail
  $('cpuavg').textContent=s.cpu_avg.toFixed(0)+'%';$('ncpu').textContent=s.cpu.length;
  $('cores').innerHTML=s.cpu.map(v=>`<div class="core" title="${v.toFixed(0)}%"><i style="height:${Math.max(v,2)}%"></i></div>`).join('');
  const h=s.cpu_hist;if(h.length>1){const pts=h.map((v,i)=>`${(i*240/(h.length-1)).toFixed(1)},${(60-v*.56).toFixed(1)}`).join(' ');
   $('cpuhist').innerHTML=`<polyline points="0,60 ${pts} 240,60" fill="var(--accent-soft)" stroke="none"/><polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.6"/>`;}
  $('mem').textContent=s.mem_pct.toFixed(0)+'%';$('memtxt').textContent=s.mem_gb+' of '+s.mem_total+' GB';$('membar').style.width=s.mem_pct+'%';
  const live=jobs.filter(j=>j.state==='running'&&j.pid);
  $('procs').innerHTML=live.length?live.map(j=>`<div class="proc"><span>${j.name}</span><span class="mono">${dur(j.elapsed)}</span></div>`).join(''):'none';
}
async function tick(){try{render(await (await fetch('/status',{cache:'no-store'})).json())}catch(e){$('clock').textContent='monitor offline'}}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body, ctype = json.dumps(SNAPSHOT).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    threading.Thread(target=_sampler, daemon=True).start()
    threading.Thread(target=_refresher, daemon=True).start()
    print(f"monitor on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
