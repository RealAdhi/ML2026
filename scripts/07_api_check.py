"""Call the running API for every served trip and print a results table (also saved to results/api_check.csv).

usage: python scripts/07_api_check.py [http://localhost:8000]
"""
import json
import os
import sys
import urllib.request

import pandas as pd

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


print(call("/health"))
rows = []
for t in call("/trips")["trips"]:
    r = call("/predict", {"trip_id": t["trip_id"]})
    m = r["metrics"]
    rows.append(dict(trip=r["trip_id"], n_obs=r["n_obs"], clf_acc=m["clf"]["acc"], hmm_acc=m["hmm"]["acc"],
                     clf_cont=m["clf"]["continuity"], hmm_cont=m["hmm"]["continuity"], hmm_route_f1=m["hmm"]["route_f1"],
                     server_s=r["timing"]["total_s"]))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "api_check.csv"), index=False)
print(df.round(4).to_string(index=False))
