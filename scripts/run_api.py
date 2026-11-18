"""Start the PathSnap API for the two-laptop demo and print the addresses Laptop 2 needs.

usage (repo root):  python scripts/run_api.py [--port 8000] [--data DIR]
  --data   folder of trips to serve (same layout as student_test/); default ./student_test  (sets EVAL_DATA_DIR)
"""
import argparse
import os
import socket
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def lan_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # the address used to reach the outside world is usually the Wi-Fi / Ethernet one
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    env = dict(os.environ)
    if a.data:
        env["EVAL_DATA_DIR"] = os.path.abspath(a.data)
    print("=" * 66)
    print(" PathSnap API  (Laptop 1)   serving:", env.get("EVAL_DATA_DIR", os.path.join(ROOT, "student_test")))
    for ip in lan_ips():
        print(f"   health check from Laptop 2 :  http://{ip}:{a.port}/health")
        print(f"   open the UI on Laptop 2     :  http://localhost:5500/?api=http://{ip}:{a.port}")
    print("   if Laptop 2 cannot connect, allow inbound TCP", a.port, "in Windows Firewall (see docs/DEMO_RUNBOOK.md)")
    print("=" * 66, flush=True)
    sys.exit(subprocess.call([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(a.port)],
                             cwd=ROOT, env=env))
