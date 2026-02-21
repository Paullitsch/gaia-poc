#!/bin/bash
# GAIA Job Analyzer — called by cron, outputs JSON summary of new completed jobs
# Usage: bash analyze.sh → outputs new job results as JSON lines

TOKEN="gaia2026"
API="http://localhost:7434"
TRACKER="/root/.openclaw/workspace/gaia-poc/analysis/job_tracker.json"

# Get all jobs
STATUS=$(curl -s -H "Authorization: Bearer $TOKEN" "$API/api/status")
if [ -z "$STATUS" ]; then
  echo '{"error":"Cannot reach server"}'
  exit 1
fi

# Get already analyzed job IDs
ANALYZED=$(python3 -c "import json; d=json.load(open('$TRACKER')); print(' '.join(d.get('analyzed_jobs',[])))" 2>/dev/null)

# Find new completed jobs
python3 -c "
import json, sys, urllib.request
from datetime import datetime

status = json.loads('''$STATUS''')
analyzed = set('$ANALYZED'.split())
tracker = json.load(open('$TRACKER'))
new_results = []

for job in status.get('jobs', []):
    jid = job['id']
    if job['status'] != 'completed' or jid in analyzed:
        continue
    
    # Fetch results
    try:
        req = urllib.request.Request(f'$API/api/results/{jid}',
            headers={'Authorization': 'Bearer $TOKEN'})
        resp = json.load(urllib.request.urlopen(req))
        rows = resp.get('results', [])
        if not rows:
            continue
    except:
        continue
    
    # Extract metrics
    best_ever = max(float(r['data'].get('best_ever', '0')) for r in rows if 'best_ever' in r['data'])
    evals = max(int(r['data'].get('evals', '0')) for r in rows if 'evals' in r['data'])
    gens = max(r['generation'] for r in rows)
    time_s = max(float(r['data'].get('time', '0')) for r in rows if 'time' in r['data'])
    sigma = float(rows[-1]['data'].get('sigma', '0')) if 'sigma' in rows[-1].get('data', {}) else None
    mean = float(rows[-1]['data'].get('mean', '0')) if 'mean' in rows[-1].get('data', {}) else None
    
    env = job['environment']
    method = job['method']
    threshold = 200 if 'Lunar' in env else 300
    solved = best_ever >= threshold
    is_backprop = 'ppo' in method
    
    # Plausibility checks
    issues = []
    if best_ever == 0 and evals > 1000:
        issues.append('zero_score_suspicious')
    if gens < 3:
        issues.append('too_few_generations')
    if sigma is not None and sigma < 1e-10 and not solved:
        issues.append('sigma_collapsed')
    if sigma is not None and sigma > 10:
        issues.append('sigma_exploded')
    if mean is not None and abs(mean) > 10000:
        issues.append('extreme_mean')
    if evals > 0 and time_s > 0:
        evals_per_sec = evals / time_s
        if evals_per_sec < 1:
            issues.append('extremely_slow')
    
    result = {
        'job_id': jid[:8],
        'method': method,
        'environment': env,
        'best_ever': round(best_ever, 1),
        'evals': evals,
        'generations': gens,
        'time_s': round(time_s, 1),
        'sigma': round(sigma, 6) if sigma else None,
        'solved': solved,
        'is_backprop': is_backprop,
        'threshold': threshold,
        'issues': issues,
        'params': job.get('params', {}),
    }
    new_results.append(result)
    tracker['analyzed_jobs'].append(jid)

tracker['last_check'] = datetime.utcnow().isoformat()
json.dump(tracker, open('$TRACKER', 'w'), indent=2)

# Output new results as JSON
if new_results:
    print(json.dumps({'new_jobs': len(new_results), 'results': new_results}, indent=2))
else:
    print(json.dumps({'new_jobs': 0}))
"
