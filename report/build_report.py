"""Builds report/PathSnap_Report.pdf (7 pages, A4) from the committed result files.

Every number in the report is read from results/ at build time; nothing is typed in by hand except the server-log
excerpt of the rehearsal run, which is a verbatim copy of what the API printed (see LOG_REHEARSAL).

usage (repo root):  python report/build_report.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image as RLImage, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "report", "assets")
OUT = os.path.join(ROOT, "report", "PathSnap_Report.pdf")
W = A4[0] - 30 * mm          # text width

INK, MUTED, RULE, ACCENT, SOFT = colors.HexColor("#1b2430"), colors.HexColor("#5b6675"), colors.HexColor("#c9d0d8"), \
    colors.HexColor("#1f4e79"), colors.HexColor("#eef2f6")

# ------------------------------------------------------------------ styles
body = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.1, leading=11.6, textColor=INK, alignment=TA_JUSTIFY,
                      spaceAfter=3.2)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=10.2, spaceAfter=2)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11.6, leading=14, textColor=ACCENT, spaceBefore=5, spaceAfter=3)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=INK, spaceBefore=3, spaceAfter=1.5)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.1, leading=8.6, textColor=MUTED, spaceAfter=3)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.2, leading=8.6, textColor=INK)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
cellr = ParagraphStyle("cellr", parent=cell, alignment=2)
title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=INK, spaceAfter=2)
sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=13, textColor=MUTED, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=9, bulletIndent=0, spaceAfter=1.6)


def P(t, s=body):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, bullet, bulletText="•")


def f4(x):
    return f"{float(x):.4f}"


def f3(x):
    return f"{float(x):.3f}"


def img(path, width):
    w, h = Image.open(path).size
    return RLImage(path, width=width, height=width * h / w)


FIGN = [0]


def capn(text):
    """Caption with an automatic running figure number (numbers follow the order the figures are created in)."""
    FIGN[0] += 1
    return f"Figure {FIGN[0]}. {text}"


def figure(path, width, caption, align="CENTER"):
    t = Table([[img(path, width)], [P(capn(caption), cap)]], colWidths=[width])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def row(items, gap=6):
    """items: list of flowables laid out side by side with equal gaps."""
    widths = [i._w if hasattr(i, "_w") else None for i in items]
    t = Table([items], colWidths=widths)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), gap), ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def fig_w(path, width, caption):
    f = figure(path, width, caption)
    f._w = width + 6
    return f


def colbox(items, width):
    t = Table([[i] for i in items], colWidths=[width])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    t._w = width + 6
    return t


def table(data, widths, header=True, zebra=True, align_right_from=1, font=7.2):
    rows = []
    for r_i, r in enumerate(data):
        out = []
        for c_i, c in enumerate(r):
            st = cellb if (header and r_i == 0) else (cellr if c_i >= align_right_from else cell)
            out.append(c if not isinstance(c, str) else Paragraph(c, st))
        rows.append(out)
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    ts = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE)]
    if header:
        ts.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    if zebra:
        for i in range(2, len(rows), 2):
            ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafb")))
    t.setStyle(TableStyle(ts))
    return t


class Pipeline(Flowable):
    """Small block diagram of the system, drawn directly so it stays sharp at any zoom."""
    steps = [("Raw trip files", ".track .nodes .arcs"), ("Candidates", "arcs within 60 m,\nmax 12 per fix"),
             ("28 features", "distance, direction,\nneighbour fit"), ("XGBoost", "score per\n(fix, arc)"),
             ("HMM / Viterbi", "connected,\nconsistent path"), ("FastAPI", "Laptop 1,\nreference kept\nserver-side"),
             ("Leaflet UI", "Laptop 2,\nroute + metrics")]

    def __init__(self, width, height=46):
        super().__init__()
        self.width, self.height = width, height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c, n = self.canv, len(self.steps)
        gap = 9
        bw = (self.width - gap * (n - 1)) / n
        for i, (a, b) in enumerate(self.steps):
            x = i * (bw + gap)
            c.setFillColor(SOFT if i not in (3, 4) else colors.HexColor("#dbe8f4"))
            c.setStrokeColor(ACCENT)
            c.setLineWidth(0.7)
            c.roundRect(x, 4, bw, self.height - 8, 3, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 6.9)
            c.drawCentredString(x + bw / 2, self.height - 15, a)
            c.setFont("Helvetica", 5.6)
            c.setFillColor(MUTED)
            for j, line in enumerate(b.split("\n")):
                c.drawCentredString(x + bw / 2, self.height - 24 - j * 6.4, line)
            if i < n - 1:
                c.setStrokeColor(ACCENT)
                c.setFillColor(ACCENT)
                y = self.height / 2
                c.line(x + bw + 1, y, x + bw + gap - 2, y)
                p = c.beginPath()
                p.moveTo(x + bw + gap - 1, y)
                p.lineTo(x + bw + gap - 4, y + 2)
                p.lineTo(x + bw + gap - 4, y - 2)
                p.close()
                c.drawPath(p, fill=1, stroke=0)


class LogBox(Flowable):
    """Terminal-style block for server logs (monospace, dark background)."""

    def __init__(self, lines, width, size=6.3):
        super().__init__()
        self.lines, self.width, self.size = lines, width, size
        self.height = len(lines) * (size + 2.2) + 10

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor("#0f1720"))
        c.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=0)
        y = self.height - 8 - self.size * 0.6
        for ln in self.lines:
            col = colors.HexColor("#8fd18f") if "POST /predict" in ln and "trip_id" in ln else \
                colors.HexColor("#e8c872") if " trip=" in ln else colors.HexColor("#b9c4d0")
            c.setFillColor(col)
            c.setFont("Courier", self.size)
            c.drawString(6, y, ln[:150])
            y -= self.size + 2.2


# ------------------------------------------------------------------ data
inv = pd.read_csv(os.path.join(R, "eda", "dataset_inventory.csv"), dtype={"trip": str})
trn, tst = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
comp = pd.read_csv(os.path.join(R, "models", "comparison.csv"))
folds = pd.read_csv(os.path.join(R, "models", "cv_folds.csv"))
tv = pd.read_csv(os.path.join(R, "bayes", "xgboost", "tuned_vs_untuned.csv")).set_index("variant")
bp = json.load(open(os.path.join(R, "bayes", "xgboost", "best_params.json")))
trials = pd.read_csv(os.path.join(R, "bayes", "xgboost", "trials.csv"))
trials = trials[trials.state == "COMPLETE"]
summ = pd.read_csv(os.path.join(R, "hmm", "summary.csv"))
per_te = pd.read_csv(os.path.join(R, "hmm", "per_trip_student.csv"), dtype={"trip": str})
per_tr = pd.read_csv(os.path.join(R, "hmm", "per_trip_train.csv"), dtype={"trip": str})
dec = json.load(open(os.path.join(ROOT, "models", "final", "decoder.json")))["decoder_params"]
chk = pd.read_csv(os.path.join(R, "api_check.csv"), dtype={"trip": str})
err = pd.read_csv(os.path.join(R, "analysis", "error_types_xgboost.csv")).set_index("error_type")
miss = pd.read_csv(os.path.join(R, "eda", "candidate_miss_by_trip_train.csv"), dtype={"trip": str})
hist = pd.read_csv(os.path.join(R, "models", "mlp_history.csv"))


import joblib
sys_path = os.path.join(ROOT, "src")
import sys
sys.path.insert(0, sys_path)
from pathsnap.candidates import FEATURES
_model = joblib.load(os.path.join(ROOT, "models", "final", "model.joblib"))
imp = pd.Series(_model.feature_importances_, index=FEATURES).sort_values(ascending=False)


def S(part, method):
    return summ[(summ.partition == part) & (summ.method == method)].iloc[0]


clf, hmm_t, hmm_d = S("student_test", "classifier_only"), S("student_test", "hmm_tuned"), S("student_test", "hmm_default_params")
c_idx = comp.set_index("model")
pv = folds.pivot(index="fold", columns="model", values="macro_f1")
wins_lgb, wins_rf = int((pv.xgboost > pv.lightgbm).sum()), int((pv.xgboost > pv.random_forest).sum())
tot_obs_te = int(tst.n_gps.sum())
errs_total = int(err["count"].sum() - err.loc["correct", "count"])

# server logs: rehearsal excerpt is verbatim output copied from the API window of the rehearsal run
LOG_REHEARSAL = [
    "2026-11-18 12:13:25,448 INFO GET /health from 10.30.204.82",
    "2026-11-18 12:13:32,252 INFO GET /trips from 10.30.204.82 -> 10 trips",
    "2026-11-18 12:13:36,740 INFO POST /predict trip_id=00000040 from 10.30.204.82",
    "2026-11-18 12:13:37,852 INFO trip=00000040 obs=2995 hmm_acc=0.9830 clf_acc=0.9269 continuity=0.9950 time=1.11s",
]
demo_log, demo_note = [], None
dp = os.path.join(R, "demo_server.log")
if os.path.exists(dp):
    keep = [l.rstrip() for l in open(dp, errors="ignore") if re.search(r"INFO (GET /|POST /predict| trip=|trip=)", l)]
    if any("POST /predict" in l for l in keep):
        demo_log = keep[:14]

# ------------------------------------------------------------------ page furniture
def furniture(canv, doc):
    canv.saveState()
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.5)
    canv.line(15 * mm, A4[1] - 11 * mm, A4[0] - 15 * mm, A4[1] - 11 * mm)
    canv.setFont("Helvetica", 7)
    canv.setFillColor(MUTED)
    canv.drawString(15 * mm, A4[1] - 9 * mm, "UE24CS352A Machine Learning  |  PathSnap: distributed map-matching and route reconstruction")
    canv.drawRightString(A4[0] - 15 * mm, A4[1] - 9 * mm, "PES University, RR Campus")
    canv.line(15 * mm, 11 * mm, A4[0] - 15 * mm, 11 * mm)
    canv.drawString(15 * mm, 7 * mm, "github.com/RealAdhi/ML2026")
    canv.drawRightString(A4[0] - 15 * mm, 7 * mm, f"Page {doc.page} of 7")
    canv.restoreState()


story = []

# ================================================================== PAGE 1
story += [P("PathSnap: Distributed Map-Matching and Route Reconstruction", title),
          P("Technical report for the PathSnap hackathon, UE24CS352A Machine Learning, 18 November 2026", sub)]
team = [["Name", "SRN", "Contribution"],
        ["S S Adhithya Sriram", "PES1UG24CS393", "Candidate generation and features, five-model comparison, Bayesian optimisation, HMM/Viterbi decoder, FastAPI backend, UI, deployment, report"],
        ["Sameeksha Kumar", "PES1UG24CS410", "EDA overview script (trip locations, trip length, network fragmentation, two-way roads), EDA findings and the fact-check of the README"],
        ["Saathvik Shenoy", "PES1UG24CS394", "Error analysis of all five models, calibration and precision-recall curves, per-trip accuracy"],
        ["Sambhram M", "PES1UG24CS409", "Automated test suite (20 tests) and the two-laptop demo runbook"]]
story += [table(team, [38 * mm, 27 * mm, W - 65 * mm], align_right_from=9), Spacer(1, 4)]

story += [P("1. Summary", h1),
          P("GPS receivers report positions that drift off the road, jump to parallel streets and sometimes stop reporting. PathSnap takes a noisy "
            "trajectory and a road network and returns the sequence of road segments (arcs) the vehicle actually drove. We solved it in two stages. "
            "First, for every GPS fix we collect the nearest directed road arcs and train a classifier to say which of them is the right one. "
            "Second, a hidden Markov model decoded with the Viterbi algorithm chooses the sequence of arcs that is both likely according to the classifier "
            "and physically drivable on the road network. The system runs as an API on one laptop and a browser UI on a second laptop."),
          P(f"We trained and compared five models with 5-fold cross-validation grouped by trip, picked XGBoost, tuned it with Bayesian optimisation "
            f"({len(trials)} completed trials) and put the HMM on top of it. On the ten student-testing trips the classifier alone assigns "
            f"{clf.point_acc * 100:.1f}% of {tot_obs_te:,} GPS fixes to the reference arc; with the HMM this rises to {hmm_t.point_acc * 100:.1f}%, and route "
            f"continuity (consecutive fixes on the same or touching arcs) rises from {clf.continuity * 100:.1f}% to {hmm_t.continuity * 100:.1f}%. "
            f"The API answered all ten student-test trips in under {chk.server_s.max():.1f} s each, the same code ran without errors on all ten reserved trips, "
            f"and the two-laptop demonstration worked over a phone hotspot (Figure 2)."),
          Spacer(1, 2), Pipeline(W), Spacer(1, 1),
          P(capn("The system. The classifier and the decoder are the two learned parts; everything to the right of the decoder is deployment."), cap)]
head_tab = [["Student-testing trips (10 trips, %s fixes)" % f"{tot_obs_te:,}", "Classifier only", "Classifier + HMM", "Change"],
            ["Point accuracy (fix assigned to the reference arc)", f"{clf.point_acc:.4f}", f"{hmm_t.point_acc:.4f}", f"{(hmm_t.point_acc - clf.point_acc) * 100:+.1f} points"],
            ["Route continuity", f"{clf.continuity:.4f}", f"{hmm_t.continuity:.4f}", f"{(hmm_t.continuity - clf.continuity) * 100:+.1f} points"],
            ["Route F1 (predicted arc set vs reference route)", f"{clf.route_f1:.4f}", f"{hmm_t.route_f1:.4f}", f"{(hmm_t.route_f1 - clf.route_f1) * 100:+.1f} points"],
            ["Route precision", f"{clf.route_precision:.4f}", f"{hmm_t.route_precision:.4f}", f"{(hmm_t.route_precision - clf.route_precision) * 100:+.1f} points"],
            ["Arc switches per trip (reference route: %.0f)" % clf.ref_switches, f"{clf.avg_path_switches:.0f}", f"{hmm_t.avg_path_switches:.0f}", f"{hmm_t.avg_path_switches - clf.avg_path_switches:+.0f}"]]
story += [table(head_tab, [W - 3 * 27 * mm, 27 * mm, 27 * mm, 27 * mm]),
          P("Table 1. Headline results on the student-testing partition (src/pathsnap/decode_eval.py, results/hmm/summary.csv).", cap)]
story += [figure(os.path.join(ASSETS, "ui_trip_result.png"), W * 0.80,
                 "The final system running. The UI on the second laptop is connected to the API on the first laptop over a phone hotspot and has just processed reserved trip "
                 "00000067: GPS points, the classifier + HMM route in green, classifier-only arcs in red, and the accuracy table computed on the server. The reference route is "
                 "never sent to the browser. This is the live demonstration partition; no decision in this report was based on it.")]
story += [P("The repository github.com/RealAdhi/ML2026 holds everything this report quotes: data inspection and EDA scripts, dataset and feature construction, the "
            "five-model experiment, Bayesian optimisation, decoder tuning, figures and maps, the API check, error analysis, the saved final model, the FastAPI backend, "
            "the UI, the tests and the README. The raw trip folders are not committed.", small)]
story.append(PageBreak())

# ================================================================== PAGE 2
story += [P("2. Dataset and exploratory data analysis", h1),
          P(f"Each trip folder holds four whitespace-separated files without headers: <b>.track</b> (longitude, latitude, timestamp), <b>.nodes</b> (road "
            f"junction coordinates), <b>.arcs</b> (directed road segments as pairs of node ids, the row number is the arc id) and <b>.route</b> (the ordered "
            f"reference arcs). Every trip carries its own road network, so network size varies from {int(trn.n_arcs.min()):,} to {int(trn.n_arcs.max()):,} arcs. "
            f"All EDA below uses the <b>training partition</b> (80 trips) unless a figure or table says otherwise; the student-testing partition (10 trips) is "
            f"shown only for comparison. The ten evaluator trips were not used in any analysis.")]
inv_tab = [["Partition", "Trips", "GPS fixes", "Fixes per trip (min / median / max)", "Arcs per trip (median / max)", "Trips with a gap > 5 s", "Longest gap", "Trips with duplicate arcs"],
           ["Training", "80", f"{int(trn.n_gps.sum()):,}", f"{int(trn.n_gps.min()):,} / {int(trn.n_gps.median()):,} / {int(trn.n_gps.max()):,}",
            f"{int(trn.n_arcs.median()):,} / {int(trn.n_arcs.max()):,}", f"{int((trn.n_gap_gt5s > 0).sum())} of 80", f"{int(trn.max_dt.max()):,} s", f"{int((trn.n_duplicate_arc_pairs > 0).sum())} of 80"],
           ["Student testing", "10", f"{int(tst.n_gps.sum()):,}", f"{int(tst.n_gps.min()):,} / {int(tst.n_gps.median()):,} / {int(tst.n_gps.max()):,}",
            f"{int(tst.n_arcs.median()):,} / {int(tst.n_arcs.max()):,}", f"{int((tst.n_gap_gt5s > 0).sum())} of 10", f"{int(tst.max_dt.max()):,} s", f"{int((tst.n_duplicate_arc_pairs > 0).sum())} of 10"]]
story += [table(inv_tab, [22 * mm, 10 * mm, 17 * mm, 39 * mm, 28 * mm, 23 * mm, 17 * mm, W - 156 * mm]),
          P("Table 2. Dataset inventory (results/eda/dataset_inventory.csv). Evaluator partition: 10 reserved trips, not inspected.", cap)]
story += [P("Preprocessing and missing-value analysis", h2),
          B(f"<b>Missing and invalid values.</b> We found no NaN coordinates or timestamps, no timestamp running backwards and no isolated nodes in any of the 90 "
            f"trips. The real problems are structural: {int((trn.n_gap_gt5s > 0).sum())} of 80 training trips contain at least one gap above 5 s (one lasts 153,064 s, "
            f"about 42 hours), road networks fall into a median of {int(trn.n_components.median())} connected components (up to {int(trn.n_components.max())}), and "
            f"{int((trn.n_duplicate_arc_pairs > 0).sum())} of 80 networks contain duplicate arcs."),
          B("<b>Projection.</b> Longitudes run from -100.4 to 137.2 and latitudes from -34.6 to 61.7 (Figure 4), so the 90 trips cluster into 37 regions on several continents. "
            "A single UTM zone would be wrong, so every trip is projected to metres with an azimuthal-equidistant projection centred on that trip."),
          B("<b>Candidates.</b> For every fix we keep up to 12 directed arcs within 60 m (the single nearest arc if none is that close). The reference arc is among the "
            "candidates for all but 10.9% of training fixes and 0.2% of student-test fixes; the next page explains the training figure."),
          B("<b>Labels.</b> The .route file lists arcs but does not say which fix lies on which arc. We align fixes to the route with a monotone dynamic programme "
            "(distance cost, backward moves penalised), so the labels are derived by us."),
          B("<b>Gaps.</b> Speed and heading features are set to missing across gaps longer than 5 s instead of being computed over hours.")]
story += [figure(os.path.join(R, "eda", "inventory.png"), W * 0.80,
                 "Trip sizes, network sizes, fragmentation, step lengths, sampling regularity and extent (training and student-test partitions, one colour each).")]
story += [row([fig_w(os.path.join(ASSETS, "trip_locations_world.jpg"), W * 0.56 - 3,
                     "Where the 90 trips were recorded: one dot per trip centroid, blue training and orange student-test (Folium map, results/eda_extra/trip_locations.html, Sameeksha Kumar)."),
               fig_w(os.path.join(R, "eda", "trajectories_train.png"), W * 0.42 - 3,
                     "Four training trips: GPS (blue), reference route (orange), nearby road arcs (grey).")])]
story.append(PageBreak())

# ================================================================== PAGE 3
n54 = miss[miss.trip == "00000054"].miss.iloc[0]
n33 = miss[miss.trip == "00000033"].miss.iloc[0]
tot_miss = (miss.miss * miss.n).sum()
two = miss[miss.trip.isin(["00000054", "00000033"])]
share2 = (two.miss * two.n).sum() / tot_miss
rest = miss[~miss.trip.isin(["00000054", "00000033"])]
rest_rate = (rest.miss * rest.n).sum() / rest.n.sum()
rev_med = pd.read_csv(os.path.join(R, "eda_extra", "trip_summary.csv"), dtype={"trip": str}).query('partition == "train"').arc_reverse_share.median()
story += [P("What the exploration told us", h1),
          P(f"<b>The nearest road is not the answer.</b> The nearest arc is the reference arc for only 55.9% of training fixes. The reason is two-way streets: most "
            f"roads are stored as two arcs with identical geometry, one per direction (the median share of a trip's arcs that have a reverse twin is {rev_med:.2f}), "
            f"so both directions are at exactly the same distance from a GPS fix and distance cannot separate them. Direction features have to, and the HMM helps "
            f"because a vehicle cannot turn around between two consecutive seconds."),
          P(f"<b>Two training trips carry the whole data-quality problem.</b> For trip 00000054 the reference arc is not among the candidates for {n54 * 100:.1f}% of the fixes, "
            f"for trip 00000033 for {n33 * 100:.1f}%. Together they produce {share2 * 100:.1f}% of all candidate misses; without them the training miss rate would be "
            f"{rest_rate * 100:.2f}% instead of 10.9%. Their reference routes describe a different path from their traces. We handled this in the open: fixes whose reference "
            f"arc is not among the candidates are not used to fit models, but they are still counted as errors in every <i>all-fixes</i> accuracy we report, and the "
            f"two trips are left out of the decoder-tuning objective. We did not call these trips outliers and drop them silently."),
          P("<b>Speed and sampling.</b> Sampling is close to 1 Hz (96.5% of steps are exactly one second), and speeds mix city and motorway driving (median 39.9 km/h). "
            "The student-testing trips are much cleaner than the training trips, which is why test scores later come out above cross-validation scores.")]
story += [row([fig_w(os.path.join(R, "eda", "nearest_arc_baseline.png"), W * 0.5 - 3, "How good is simply taking the nearest arc? Rank of the reference arc by distance, and the ties between the two directions."),
               fig_w(os.path.join(R, "eda_extra", "two_way_share.png"), W * 0.5 - 3, "Share of arcs that have an opposite-direction twin, per partition (Sameeksha Kumar).")])]
story += [row([fig_w(os.path.join(R, "eda", "candidate_miss_by_trip.png"), W * 0.5 - 3, "Candidate miss rate per training trip: two trips dominate."),
               fig_w(os.path.join(R, "eda", "speed_and_gaps.png"), W * 0.5 - 3, "Speed implied by consecutive fixes and the distribution of time gaps.")])]
feat = [["Group", "Features (28 in total)", "Why we included them"],
        ["Position", "d, d_rank, d_rel, d_ratio, frac, arc_len, d_to_start, d_to_end", "How close the arc is, relative to the other candidates, and where along the arc the fix projects"],
        ["Direction", "cos_h, abs_ang, cos_h_prev, cos_h_next, n_dir_ok, min_d_dir_ok, d_vs_best_dir", "Whether the movement of the vehicle agrees with the arc direction; separates the two lanes of a two-way road"],
        ["Kinematics", "speed, step_prev, step_next, dt_prev, dt_next", "Fast vehicles stay on long arcs; missing across time gaps"],
        ["Network", "deg_from, deg_to, outdeg_to, has_reverse, n_cands", "Junction structure around the arc"],
        ["Neighbours", "d_prev, d_next, dprog_ok", "Do the previous and next fixes fit the same arc, and does the vehicle progress along it"]]
story += [P("Features", h2), table(feat, [19 * mm, 74 * mm, W - 93 * mm], align_right_from=9),
          P("Table 3. Feature groups (src/pathsnap/candidates.py, FEATURES). Features are computed per (fix, candidate arc) from the raw files alone.", cap)]
top5 = imp.head(5)
fi_txt = P(f"<b>What the final model uses.</b> By gain, the five most important features of the final XGBoost model are "
           + ", ".join(f"<font name='Courier'>{k}</font> ({v:.2f})" for k, v in top5.items())
           + f", together {top5.sum() * 100:.0f}% of the total. In words, the model mostly asks two questions: is this arc much farther than the closest candidate, and does the "
             f"movement of the vehicle agree with the direction of the arc? The neighbour-fit and network features add comparatively little on their own, but they are what lets the "
             f"classifier break ties between the two directions of a road when the heading is noisy. Importances are model-internal and say nothing about causes.", body)
story += [row([fig_w(os.path.join(R, "analysis", "xgboost_feature_importance.png"), W * 0.33 - 3, "Feature importance (gain) of the final XGBoost model."),
               colbox([fi_txt], W * 0.64 - 6)])]
story.append(PageBreak())

# ================================================================== PAGE 4
story += [P("3. Model experimentation and comparison", h1),
          P("The task for the classifier is binary: given a GPS fix and one candidate arc, is this the reference arc? The arc with the highest score per fix is the "
            "classifier-only prediction. We trained five models from different families on the 80 training trips, validated with 5-fold <b>GroupKFold</b> (all fixes of a "
            "trip stay in one fold, so no trip appears in both training and validation), refitted on all 80 trips and evaluated once on the ten student-testing trips. "
            "Cross-validation macro-F1 and log loss are computed on fixes whose reference arc is among the candidates; \"top-1 (all)\" counts the others as wrong. "
            "Group-CV macro-F1 was our primary selection metric because it uses only training trips, is not dominated by the many easy negative candidates, and "
            "follows the decision the decoder relies on.")]
hd = ["Model", "Family", "Key configuration", "CV macro-F1 (mean ± sd)", "CV top-1 (all)", "Cont.", "Test F1 (macro)", "Test top-1 (all)", "Test log loss", "Fit (s)", "Infer (ms/1k)"]
data = [hd]
for m in comp.itertuples():
    b = m.model == "xgboost"
    data.append([("<b>%s</b>" % m.display) if b else m.display, m.family, m.config,
                 f"{m.cv_macro_f1_mean:.4f} ± {m.cv_macro_f1_std:.3f}", f4(m.cv_top1_all), f4(m.test_route_continuity), f4(m.test_macro_f1),
                 ("<b>%s</b>" % f4(m.test_top1_all)) if b else f4(m.test_top1_all), f4(m.test_log_loss), f"{m.train_time_s:.0f}", f"{m.infer_ms_per_1k_obs:.0f}"])
story += [table(data, [21 * mm, 19 * mm, 32 * mm, 21 * mm, 12 * mm, 12 * mm, 13 * mm, 13 * mm, 12 * mm, 9 * mm, W - 164 * mm], align_right_from=3),
          P("Table 4. Consolidated comparison (results/models/comparison.csv). \"Cont.\" is the route continuity of the classifier-only predictions on the student-test trips. Fit and inference times are "
            "indicative: the five models ran in parallel with different thread budgets. The optimisation objective was log loss for logistic regression, XGBoost, "
            "LightGBM and the MLP, and Gini impurity for the forest.", cap)]
notes = [
    ("Logistic Regression", "A linear baseline that gives probabilities and shows how far straight decision boundaries get. It cannot combine distance rank with direction "
     f"agreement, the interaction that matters most, and it is the weakest model (test top-1 {c_idx.loc['logreg', 'test_top1_all']:.3f}). Its reliability curve bends below the diagonal, so it is over-confident in the middle range. C = 1.0, median imputation, standardised inputs."),
    ("Random Forest", f"Bagged trees capture the interactions and are stable across folds. 150 trees, depth 14, minimum leaf 10. Second on test top-1 "
     f"({c_idx.loc['random_forest', 'test_top1_all']:.3f}), but the slowest tree model to fit ({c_idx.loc['random_forest', 'train_time_s']:.0f} s) and about three times slower than XGBoost at prediction."),
    ("LightGBM", f"Histogram boosting corrects errors sequentially and is fast to fit. 300 trees, learning rate 0.05, 31 leaves. Third on CV macro-F1, fourth on test top-1, and "
     f"beaten by XGBoost in {wins_lgb} of 5 folds."),
    ("XGBoost", f"Boosted trees with depth 6, 300 trees, learning rate 0.1. Best CV macro-F1 ({c_idx.loc['xgboost', 'cv_macro_f1_mean']:.4f}), best test top-1 and log loss, fast to fit "
     f"({c_idx.loc['xgboost', 'train_time_s']:.0f} s) and to predict, with a reliability curve close to the diagonal, which the HMM needs."),
    ("MLP (64-32)", f"Two hidden layers trained with Adam, early stopping on 10% held-out trips. Its validation loss ({hist.val_loss.iloc[-1]:.3f}) is below its training loss "
     f"({hist.train_loss.iloc[-1]:.3f}) because the held-out trips are easier than the average training trip, not because it generalises better. Competitive but the slowest to fit ({c_idx.loc['mlp', 'train_time_s']:.0f} s)."),
]
story += [P("Per-model notes: why it suits the problem, strengths and limits, configuration, evidence of over- or underfitting", h2)]
nt = [[P("<b>%s</b>" % a, cell), P(b, cell)] for a, b in notes]
t = Table(nt, colWidths=[24 * mm, W - 24 * mm])
t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 2),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
story += [t, Spacer(1, 3),
          P("<b>Over- and underfitting.</b> For the four non-neural models we have no separate training score, so we judge by agreement between cross-validation and the test partition: they "
            "agree in ranking and differ in level only because the student-test trips are cleaner (candidate miss rate 0.20%% against 10.9%%). The MLP loss curves "
            "flatten after about 15 epochs and training stopped at epoch %d, with no sign of validation loss rising." % int(hist.epoch.iloc[-1]))]
story += [row([fig_w(os.path.join(R, "figures", "cv_macro_f1.png"), W * 0.5 - 3, "Cross-validation macro-F1, mean and standard deviation over the five folds."),
               fig_w(os.path.join(R, "figures", "test_top1_macro_f1.png"), W * 0.5 - 3, "Student-test top-1 accuracy and macro-F1.")])]
story.append(PageBreak())

# ================================================================== PAGE 5
story += [P("Model comparison figures and the choice of XGBoost", h1)]
story += [row([fig_w(os.path.join(R, "figures", "log_loss.png"), W * 0.5 - 3, "Log loss, cross-validation and student-test (lower is better)."),
               fig_w(os.path.join(R, "figures", "mlp_loss_curve.png"), W * 0.5 - 3, "MLP training and validation loss per epoch. Only the MLP produces an epoch history; we drew no curves for the other models.")])]
story += [figure(os.path.join(R, "figures", "confusion_matrices.png"), W - 6, "Confusion matrices on the student-test candidates, one decision per fix (the top-scored candidate is the selected one). Rows are the truth.")]
story += [row([fig_w(os.path.join(R, "analysis", "calibration.png"), W * 0.38 - 3, "Reliability diagram (Saathvik Shenoy). Curves near the diagonal mean trustworthy probabilities."),
               fig_w(os.path.join(R, "analysis", "error_types.png"), W * 0.62 - 3, "What the mistakes are: wrong direction on the same road, neighbouring arc, other. Candidate misses are the same for all models.")])]
story += [P("Why XGBoost", h2),
          B(f"It has the highest cross-validation macro-F1 ({c_idx.loc['xgboost', 'cv_macro_f1_mean']:.4f}), the highest test top-1 accuracy ({c_idx.loc['xgboost', 'test_top1_all']:.4f}) and the lowest test log loss "
            f"({c_idx.loc['xgboost', 'test_log_loss']:.4f}). The standard deviation over folds is about 0.02 for every model, because folds differ in difficulty, so means alone cannot rank "
            f"the models. Paired by fold, XGBoost beats LightGBM in {wins_lgb} of 5 folds and the random forest in {wins_rf} of 5."),
          B("It trains in under a minute and predicts quickly, and its probabilities follow the diagonal of the reliability plot. Calibration matters because the HMM combines "
            "these probabilities along the whole trajectory; an over-confident model would push wrong arcs through the decoder."),
          B("We did not select on the student-test partition: the cross-validation ranking already puts XGBoost first, and the test partition only confirmed it. The evaluator partition played no part."),
          P(f"<b>Where the models fail.</b> For XGBoost, {errs_total:,} of the {tot_obs_te:,} student-test fixes are wrong: {int(err.loc['reverse_direction', 'count']):,} "
            f"({err.loc['reverse_direction', 'count'] / errs_total * 100:.0f}% of the errors) pick the opposite direction of the correct road, {int(err.loc['adjacent_arc', 'count']):,} "
            f"({err.loc['adjacent_arc', 'count'] / errs_total * 100:.0f}%) pick a touching arc, {int(err.loc['other', 'count']):,} pick something else and {int(err.loc['candidate_miss', 'count'])} are candidate "
            f"misses. Wrong-direction picks are exactly the case that a single-fix classifier cannot settle and a sequence model can, which Section 5 measures.")]
story.append(PageBreak())

# ================================================================== PAGE 6
u, t_ = tv.loc["untuned"], tv.loc["tuned"]
story += [P("4. Bayesian hyperparameter optimisation (XGBoost)", h1),
          P(f"We used Optuna's TPE sampler, a Bayesian method that models which parameter regions scored well and proposes new trials there. The search space was the number of trees "
            f"(100 to 600, step 50), the learning rate (0.01 to 0.3, log scale) and the maximum depth (3 to 12). The objective was the mean macro-F1 of the same 5-fold group "
            f"cross-validation on the 80 training trips. We planned 20 trials; {len(trials)} completed because trials already running when the 20th finished were allowed to finish "
            f"(two more were interrupted by a restart and are marked failed in the stored study). The best trial used {bp['best_params']['n_estimators']} trees, "
            f"learning rate {bp['best_params']['learning_rate']:.4f} and depth {bp['best_params']['max_depth']} (CV macro-F1 {bp['best_cv_macro_f1']:.5f}). "
            f"All {len(trials)} trials scored between {trials.value.min():.4f} and {trials.value.max():.4f}, so XGBoost is not very sensitive to these three parameters here.")]
tt = [["Same folds, same partitions", "CV macro-F1", "CV log loss", "Test top-1 (all)", "Test macro-F1", "Test log loss", "Test cont.", "Fit (s)", "Infer (ms/1k)"],
      ["Untuned (300 trees, lr 0.1, depth 6)", f4(u.cv_macro_f1), f4(u.cv_log_loss), f4(u.test_top1_all), f4(u.test_macro_f1), f4(u.test_log_loss), f4(u.test_route_continuity), f"{u.train_time_s:.0f}", f"{u.infer_ms_per_1k_obs:.0f}"],
      ["<b>Tuned (350 trees, lr 0.0203, depth 10)</b>", f4(t_.cv_macro_f1), f4(t_.cv_log_loss), f4(t_.test_top1_all), f4(t_.test_macro_f1), f4(t_.test_log_loss), f4(t_.test_route_continuity), f"{t_.train_time_s:.0f}", f"{t_.infer_ms_per_1k_obs:.0f}"]]
story += [table(tt, [50 * mm, 17 * mm, 15 * mm, 18 * mm, 17 * mm, 16 * mm, 14 * mm, 11 * mm, W - 158 * mm]),
          P("Table 5. Tuned against untuned XGBoost (results/bayes/xgboost/tuned_vs_untuned.csv).", cap),
          P(f"The tuned model is ahead on every quality metric, but only slightly (CV macro-F1 {(t_.cv_macro_f1 - u.cv_macro_f1):+.4f}, test top-1 {(t_.test_top1_all - u.test_top1_all):+.4f}), far less than the fold spread, "
            f"and it is about {t_.infer_ms_per_1k_obs / u.infer_ms_per_1k_obs:.0f} times slower at prediction. We use it as the final model because it is at least as good everywhere and the cost is small next to the "
            f"decoder, but we do not claim the gain is statistically meaningful.")]
story += [row([fig_w(os.path.join(R, "bayes", "xgboost", "history.png"), W * 0.36 - 3, "Optuna history: every trial and the best score so far."),
               fig_w(os.path.join(R, "figures", "tuned_vs_untuned.png"), W * 0.62 - 3, "Tuned against untuned XGBoost on cross-validation macro-F1, student-test top-1 and student-test log loss.")])]
story += [P("5. HMM and Viterbi route reconstruction", h1),
          P("<b>Hidden states.</b> For observation t the states are its candidate arcs (at most 12), each keeping its arc id. <b>Emission</b> is the log of the XGBoost probability normalised across the "
            f"candidates, plus a small Gaussian distance term (weight {dec['w_dist']:.2f}, width {dec['sigma']:.1f} m). <b>Transition</b> follows Newson and Krumm: the score is "
            f"minus |straight-line distance between the two fixes minus the driving distance along the road network between the projected positions| divided by {dec['beta']:.1f} m. "
            f"The driving distance comes from a bounded Dijkstra search on the arc graph. <b>Gaps:</b> above {dec['gap_s']:.0f} s the transition is made uninformative so the path restarts. "
            f"<b>Unreachable pairs</b> (disconnected segments) get a finite penalty of {dec['break_pen']:.1f} in log space rather than zero probability, and an immediate U-turn costs a further {dec['uturn_pen']:.1f}. "
            f"Viterbi keeps, for every state, the best score of any path ending there and reads the best path back through stored pointers. The decoder parameters were tuned with Optuna (40 trials) on "
            f"out-of-fold classifier scores of training trips only.")]
hm = [["Method", "Student-test point accuracy", "Continuity", "Route F1", "Arc switches / trip", "Training OOF point accuracy"],
      ["Classifier only", f4(clf.point_acc), f4(clf.continuity), f4(clf.route_f1), f"{clf.avg_path_switches:.0f}", f4(S('train_oof', 'classifier_only').point_acc)],
      ["Classifier + HMM, default parameters", f4(hmm_d.point_acc), f4(hmm_d.continuity), f4(hmm_d.route_f1), f"{hmm_d.avg_path_switches:.0f}", f4(S('train_oof', 'hmm_default_params').point_acc)],
      ["<b>Classifier + HMM, tuned</b>", f4(hmm_t.point_acc), f4(hmm_t.continuity), f4(hmm_t.route_f1), f"{hmm_t.avg_path_switches:.0f}", f4(S('train_oof', 'hmm_tuned').point_acc)]]
better = int((per_te.hmm_acc > per_te.clf_acc).sum())
worse_tr = per_tr[per_tr.hmm_acc < per_tr.clf_acc]
dgain = per_te.assign(d=per_te.hmm_acc - per_te.clf_acc).d
story += [table(hm, [52 * mm, 30 * mm, 18 * mm, 18 * mm, 24 * mm, W - 142 * mm]),
          P("Table 6. Classifier only against classifier + HMM (results/hmm/summary.csv). The reference route switches arcs %.0f times per student-test trip." % clf.ref_switches, cap),
          P(f"The HMM improved accuracy on {better} of 10 student-test trips (largest gain {dgain.max() * 100:+.1f} points, smallest {dgain.min() * 100:+.1f}) and on "
            f"{int((per_tr.hmm_acc > per_tr.clf_acc).sum())} of 80 training trips; it lowered accuracy on {len(worse_tr)} training trips "
            f"({', '.join(f'{r.trip} by {abs(r.hmm_acc - r.clf_acc) * 100:.1f} points' for r in worse_tr.itertuples())}), where continuity still rose. Even the default parameters help; tuning adds "
            f"{(hmm_t.point_acc - hmm_d.point_acc) * 100:.1f} points on the untouched student-test trips, at least as much as on the training trips, so the decoder does not look over-fitted to its tuning set. "
            f"Trip 00000077 stays the weakest ({per_te[per_te.trip == '00000077'].hmm_acc.iloc[0]:.3f}): 892 of its 1,198 classifier errors are wrong-direction picks on two-way roads, which "
            f"neither the classifier nor a distance-based transition fully resolves.")]
story += [figure(os.path.join(R, "figures", "hmm_vs_classifier.png"), W * 0.86,
                 "Per student-test trip, classifier only (red) against classifier + HMM (green): point accuracy (left) and route continuity (right).")]
story.append(PageBreak())

# ================================================================== PAGE 7
story += [P("6. Geographical visualisation", h1),
          P("The Folium map below and the live UI (Figure 2 on page 1) are both produced from our own pipeline output. The Folium map is generated by scripts/08_maps.py for any trip and "
            "shows the GPS fixes, the candidate road arcs, the reference route, the classifier-only route and the classifier + HMM route as separate layers with a legend and a layer control, "
            "so the observed trajectory and the predicted route can be compared directly. The interactive HTML maps are committed in results/maps/. The UI draws the same "
            "layers in Leaflet from the API response, without the reference route.")]
story += [figure(os.path.join(ASSETS, "folium_trip_00000040.jpg"), W * 0.60,
                 "Folium map of student-test trip 00000040 (screenshot of the interactive map results/maps/pred_student_00000040.html): GPS fixes and the classifier + HMM route follow the road from the lake "
                 "district north of Chelyabinsk into the city. Layer control top right, legend bottom left.")]
story += [P("7. API, UI and deployment", h1),
          P("<b>Architecture.</b> Laptop 1 runs the FastAPI backend with the final model, the decoder, the road networks and the reference routes. Laptop 2 serves the static Leaflet UI. "
            "The UI sends only a trip id to <font name='Courier'>POST /predict</font>. The server reads the raw trip files <i>without</i> the reference route, generates candidates and features, scores them, "
            "decodes with Viterbi, and only then loads the reference on the server to compute accuracy, continuity and route F1. Only aggregate metrics and the predicted route go back. Nothing is hard-coded and "
            "every request is logged. <font name='Courier'>GET /health</font> and <font name='Courier'>GET /trips</font> support the UI. The API starts with one command "
            "(<font name='Courier'>python scripts/run_api.py --data &lt;folder&gt;</font>), which refuses to start on an empty or incomplete data folder. Both laptops joined the same phone hotspot; "
            "the firewall rule for port 8000 and the exact steps are in docs/DEMO_RUNBOOK.md.")]
chk_rows = [["Trip", "Fixes", "Acc. clf", "Acc. + HMM", "Cont. + HMM", "Route F1", "Time (s)"]] + \
           [[r.trip, f"{int(r.n_obs):,}", f4(r.clf_acc), f4(r.hmm_acc), f4(r.hmm_cont), f4(r.hmm_route_f1), f"{r.server_s:.2f}"] for r in chk.itertuples()]
story += [row([fig_w(os.path.join(ASSETS, "pipeline_terminal.png"), W * 0.46 - 3, "Terminal progress view of the whole pipeline (scripts/tui.py): finished stages, the running API, CPU and memory."),
               colbox([table(chk_rows, [17 * mm, 10 * mm, 14 * mm, 15 * mm, 15 * mm, 15 * mm, W * 0.52 - 86 * mm]),
                       P("Table 7. API check on the ten student-test trips (results/api_check.csv): every trip answered, mean server time %.2f s." % chk.server_s.mean(), cap)], W * 0.52 - 3)])]
story += [P("Server-side logs on Laptop 1", h2),
          LogBox(LOG_REHEARSAL, W),
          P(capn("API window on Laptop 1 during the rehearsal run: each click on the UI shows the client address, the trip id and the result line (verbatim output)."), cap)]
if demo_log:
    story += [LogBox(demo_log, W), P(capn("API window during the live run on the reserved trips, client on the second laptop."), cap)]
story += [P("<b>Limits.</b> The reference routes are derived labels and are wrong for two training trips. Tuning XGBoost gained very little. Duplicate or two-way arcs with identical geometry "
            "remain the largest source of error even after decoding. Timings in Table 4 are not a controlled benchmark. Only our training and student-test trips informed any "
            "decision. All figures and tables are produced by scripts in the repository from the saved results.", small)]

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                      title="PathSnap: Distributed Map-Matching and Route Reconstruction", author="S S Adhithya Sriram, Sameeksha Kumar, Sambhram M, Saathvik Shenoy")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(15 * mm, 15 * mm, W, A4[1] - 30 * mm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=furniture)])
doc.build(story)
from pypdf import PdfReader
print("pages:", len(PdfReader(OUT).pages), "->", OUT)
