# PathSnap two-laptop demo runbook

This guide is for the evaluator demonstration. The required setup is the API on Laptop 1 and the browser UI on Laptop 2. The one-laptop option near the end is only a fallback if the network fails.

## 1. Before demo day

- [ ] Both laptops are charged and have their power supplies.
- [ ] Both laptops are on the same Wi-Fi network, or Laptop 1's hotspot is ready.
- [ ] The repository is cloned and `pip install -r requirements.txt` has completed on both laptops.
- [ ] `models/final/model.joblib` is present on Laptop 1.
- [ ] The folder to serve is known. It must contain trip subfolders with `.track`, `.nodes`, `.arcs`, and `.route` files. The default is `student_test/`.
- [ ] A browser is ready on Laptop 2.
- [ ] A screenshots folder has been created for the report.

## 2. Find Laptop 1's IP address

1. On Laptop 1, open Command Prompt and run `ipconfig`.
2. Find the network adapter currently connected to the shared Wi-Fi or hotspot.
3. Read its `IPv4 Address` line. Do not use the address under an inactive adapter.
4. Keep this address handy; Laptop 2 will use it to reach the API.

## 3. Allow the API through the Windows firewall

On Laptop 1, open PowerShell **as Administrator** and add the inbound rule:

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Remove the rule after the demo from an Administrator PowerShell window:

```powershell
Remove-NetFirewallRule -DisplayName "PathSnap API 8000"
```

## 4. Start and check the API

Run these commands from the repository root on Laptop 1. Set `EVAL_DATA_DIR` only when serving a folder other than `student_test/`.

PowerShell:

```powershell
$env:EVAL_DATA_DIR = "C:\path\to\evaluator_trips"  # optional; default is ./student_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For Command Prompt (`cmd.exe`):

```bat
set EVAL_DATA_DIR=C:\path\to\evaluator_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Leave this command window open so the evaluator can see API logs. On Laptop 2, first open `http://<LAPTOP-1-IP>:8000/health` in a browser. The page should show JSON containing `"status":"ok"`.

## 5. Start the UI on Laptop 2

From the repository root on Laptop 2, run:

```bash
python -m http.server 5500 --directory ui
```

Open `http://localhost:5500/?api=http://<LAPTOP-1-IP>:8000`. The `?api=` parameter fills in the API address; the other option is to type the address into the API box and click **Connect**.

## 6. Demo script

1. Open the UI and connect it to Laptop 1's API.
2. Point out that the connected UI lists ten reserved trips.
3. Choose a trip and wait for the prediction to finish.
4. Point out the GPS points on the map.
5. Turn on or identify the predicted route layer.
6. Show the classifier-only route layer and explain that it is the sequence before HMM decoding.
7. Show the accuracy and continuity results in the table.
8. Switch to Laptop 1 and show the command window's API logs.
9. Repeat with other trips as time allows; the implementation must work for all ten.

## 7. What the evaluator should see in the API window

Each prediction should log a request line beginning `POST /predict trip_id=...`, followed by a result line shaped like:

```text
trip=... obs=... hmm_acc=... clf_acc=... continuity=... time=...s
```

**Real dry-run example:** pending the lead's ready message and an authorized two-laptop run. No example is included yet because a real log line must come from that run.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| UI says “connection failed” | Wrong API address or API process is stopped | Check the address in the UI and confirm Uvicorn is still running on Laptop 1. |
| `/health` times out on Laptop 2 | Laptops are on different networks, firewall blocks port 8000, or the IP is wrong | Recheck the active `IPv4 Address`, shared network, and firewall rule. |
| `ModuleNotFoundError` | Dependencies were not installed in the active Python environment | Activate the intended environment, then run `pip install -r requirements.txt`. |
| Port 8000 is already in use | Another process is listening on the API port | Stop that process or identify and stop the old API server, then retry. |
| Trip list is empty | `EVAL_DATA_DIR` points to a missing or incorrectly structured folder | Check the environment variable and verify each trip folder contains the expected files. |
| Map is blank but results appear | Map tile provider cannot be reached, often because there is no internet connection | Restore internet access for tiles; use the results table and API output while reconnecting. |
| First prediction is very slow | Model loading or first-time pipeline work is still running | Wait for the first request to finish, then try another trip and check the server output. |
| Browser console reports a CORS error | Browser is calling a different API address or an older API process is running | Confirm the UI address and restart the current API from this checkout. |

## 9. Backup plan

If the network between laptops fails, run the API and UI on one laptop and connect the UI to `http://localhost:8000`. This is a troubleshooting fallback; the event rules require the two-laptop arrangement for the actual evaluation.

## 10. After the demo

1. Stop the API and UI servers with `Ctrl+C` in their command windows.
2. Remove the temporary firewall rule using the command in section 3.
3. Do not commit evaluator data to GitHub.

## 11. Dry-run record

Part 3 is pending the lead's ready message. After that run, add the real API log line in section 7, record all ten results, and save the four requested screenshots under `docs/screenshots/`.

| Trip | Worked? (yes/no) | Time (s) |
|---|---|---|
| Pending dry-run | — | — |
