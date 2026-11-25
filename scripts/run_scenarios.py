"""Run acceptance scenarios and persist observed pass/fail/skip outcomes."""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/scenarios'
EXPECTED = {
    '01': 'Prepare, train, evaluate both directions and write a labelled synthetic report',
    '02': 'Missing scenes produce an actionable error and CLI status 2',
    '03': 'Invalid paths, thresholds, patch sizes and batch sizes are rejected',
    '04': 'All-cloud optical patches rejected with cloud_fraction reason',
    '05': 'Time gaps beyond seven days rejected with time_gap reason',
    '06': 'Cross-split overlapping footprints rejected even with falsified block IDs',
    '07': 'Corrupt raster and missing bands raise useful errors and write traceback',
    '08': 'Small encoder forward pass works entirely on CPU',
    '09': 'Small CUDA forward pass works if CUDA available; otherwise explicitly skipped',
    '10': 'One-epoch checkpoint resumes to epoch two with preserved history',
    '11': 'Empty gallery rejected with a clear ValueError',
    '12': 'Unexpected exception writes traceback and yields concise CLI status 1',
}

if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    cmd = [sys.executable, '-m', 'pytest', 'tests/test_scenarios.py', '-q', f'--junitxml={OUTPUT / "results.xml"}']
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    (OUTPUT / 'console.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    rows = []
    for case in ET.parse(OUTPUT / 'results.xml').iter('testcase'):
        name = case.attrib['name']
        if not name.startswith('test_scenario_'):
            continue
        number = name.split('_')[2]
        error = case.find('failure')
        if error is None:
            error = case.find('error')
        skipped = case.find('skipped')
        status = 'FAIL' if error is not None else 'SKIP' if skipped is not None else 'PASS'
        observed = error.attrib.get('message', '') if error is not None else skipped.attrib.get('message', '') if skipped is not None else 'All scenario assertions passed on actual execution'
        rows.append({'scenario': number, 'test': f'tests/test_scenarios.py::{name}', 'expected': EXPECTED[number], 'observed': observed, 'status': status, 'elapsed_s': float(case.attrib.get('time', 0))})
    (OUTPUT / 'results.json').write_text(json.dumps({'command': cmd, 'returncode': result.returncode, 'scenarios': rows}, indent=2), encoding='utf-8')
    print(result.stdout)
    print(json.dumps(rows, indent=2))
    raise SystemExit(result.returncode)
