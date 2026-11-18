# Test findings

Run these commands from the repository root:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q
```

For the non-API tests on this checkout, the final pytest line was:

```text
16 passed, 5 warnings in 1.67s
```

The four API tests did not complete in this environment: even the `/health` test remained at `tests/test_api.py::test_health` for more than 30 seconds. The test process was stopped; no API pass result is claimed. The full `python -m pytest tests -q` run therefore remains to be completed. The five warnings are Pandas deprecation warnings from `candidates.py:170` when candidate generation calls `DataFrame.drop(..., inplace=True)`.
