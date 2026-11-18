"""PathSnap terminal progress view (rich): running jobs on top, finished jobs at the bottom, refreshed every second.

usage: python scripts/tui.py          (Ctrl+C to quit)
Uses the same job discovery as scripts/monitor.py (processes, logs and output files), so anything started under
scripts/ shows up here automatically.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import monitor  # noqa: E402  (job discovery shared with the web dashboard)
import psutil  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress_bar import ProgressBar  # noqa: E402
from rich.spinner import Spinner  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

STATE_ORDER = {"running": 0, "failed": 1, "waiting": 2, "done": 3}
COLOR = {"running": "yellow", "failed": "red", "waiting": "grey50", "done": "green"}
BLOCKS = " ▁▂▃▄▅▆▇█"

LATEST = {"status": None, "err": None}


def refresher():
    last_fetch = 0.0
    while True:
        try:
            if time.time() - last_fetch > 60:
                last_fetch = time.time()
                threading.Thread(target=monitor.refresh_handoffs, daemon=True).start()
            LATEST["status"] = monitor.status()
        except Exception as e:  # never let a malformed file kill the display
            LATEST["err"] = str(e)[:80]
        time.sleep(1)


def dur(s):
    if s is None:
        return ""
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s" if s < 3600 else f"{s // 3600}h{(s % 3600) // 60:02d}m"


def metric_text(j):
    """One short line of real numbers for a job."""
    k = j["kind"]
    if j["key"].startswith("model:"):
        f = j.get("folds") or []
        t = j.get("test")
        s = f"folds {len(f)}/5"
        if f:
            s += f"  last F1 {f[-1]['f1']:.4f}"
        if t:
            s += f"  | test top-1 {t['top1']:.4f}  F1 {t['f1']:.4f}  logloss {t['ll']:.4f}"
        return s
    if k in ("tuning", "decoder"):
        tr = j.get("trials") or []
        s = f"trial {len(tr)}/{j.get('n_total')}" if j["state"] != "done" else f"{len(tr)} trials"
        if j.get("best") is not None:
            s += f"  best {j['best']:.5f}"
        t = (j.get("tuned") or {}).get("tuned")
        if t:
            s += f"  | tuned CV F1 {float(t['cv_macro_f1']):.4f}  test top-1 {float(t['test_top1_all']):.4f}"
        return s
    return j.get("detail") or ""


def render():
    st = LATEST["status"]
    if not st:
        return Panel("starting...")
    jobs = sorted(st["jobs"], key=lambda j: (STATE_ORDER[j["state"]], j["rank"]))
    n = len(jobs)
    cnt = {k: sum(j["state"] == k for j in jobs) for k in STATE_ORDER}

    tbl = Table(expand=True, box=None, pad_edge=False, header_style="bold dim")
    tbl.add_column("", width=2)
    tbl.add_column("Stage", ratio=3, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Progress", width=26)
    tbl.add_column("Detail", ratio=5, overflow="ellipsis", no_wrap=True)
    tbl.add_column("Time", width=8, justify="right")
    last = None
    for j in jobs:
        if j["state"] != last:
            label = {"running": "IN PROGRESS", "failed": "NEEDS ATTENTION", "waiting": "QUEUED", "done": "COMPLETED"}[j["state"]]
            tbl.add_row("", Text(label, style=f"bold {COLOR[j['state']]}"), "", "", "")
            last = j["state"]
        c = COLOR[j["state"]]
        icon = Spinner("dots", style=c) if j["state"] == "running" else Text({"done": "✔", "failed": "✘", "waiting": "·"}[j["state"]], style=c)
        pct = 100 if j["state"] == "done" else (j.get("pct") or 0)
        prog = Table.grid(expand=True)
        prog.add_column(ratio=1)
        prog.add_column(width=5, justify="right")
        prog.add_row(ProgressBar(total=100, completed=pct, width=19, complete_style=c, finished_style="green", style="grey23"),
                     Text(f"{pct:.0f}%", style=c))
        dim = "dim" if j["state"] == "done" else ""
        tbl.add_row(icon, Text(j["name"], style=f"{dim} bold" if j["state"] == "running" else dim), prog,
                    Text(metric_text(j), style=dim), Text(dur(j.get("elapsed")), style=dim))

    cpu = st["cpu"]
    cpu_txt = Text()
    for v in cpu:
        cpu_txt.append(BLOCKS[min(8, int(v / 100 * 8.99))], style="green" if v < 50 else "yellow" if v < 85 else "red")
    sysg = Table.grid(padding=(0, 2))
    sysg.add_row(Text("CPU ", style="bold"), cpu_txt, Text(f"{st['cpu_avg']:.0f}% avg", style="bold"))
    sysg.add_row(Text("RAM ", style="bold"),
                 ProgressBar(total=100, completed=st["mem_pct"], width=len(cpu), complete_style="cyan"),
                 Text(f"{st['mem_gb']} / {st['mem_total']} GB"))

    chain = []
    cp = os.path.join(monitor.RES, "chain.log")
    if os.path.exists(cp):
        chain = [l for l in open(cp, errors="ignore").read().splitlines() if l.startswith("[chain]")][-4:]
    head = Text.assemble((f" {cnt['done']}/{n} stages complete ", "bold green"), ("  "), (f"{cnt['running']} running ", "bold yellow"),
                         ("  "), (f"{cnt['failed']} failed ", "bold red" if cnt["failed"] else "dim"), ("  "),
                         (time.strftime("%H:%M:%S"), "dim"))
    total_bar = ProgressBar(total=n, completed=cnt["done"], width=None, complete_style="green", finished_style="green")
    parts = [head, total_bar, Text(""), tbl, Text(""), sysg]
    if chain:
        parts += [Text(""), Text("pipeline chain:", style="bold dim")] + [Text("  " + c, style="dim") for c in chain]
    return Panel(Group(*parts), title="[bold]PathSnap pipeline[/bold]", subtitle="Ctrl+C to quit", border_style="cyan")


if __name__ == "__main__":
    threading.Thread(target=monitor._sampler, daemon=True).start()
    threading.Thread(target=refresher, daemon=True).start()
    try:
        with Live(render(), refresh_per_second=8, screen=False, transient=False) as live:
            while True:
                live.update(render())
                time.sleep(0.125)
    except KeyboardInterrupt:
        pass
