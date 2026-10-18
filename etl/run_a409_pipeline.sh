#!/bin/bash
# Research-machine pipeline (A409): wait for the raw data and the pinned LEAN image, rebuild the LEAN data,
# reproduce one workstation result as an environment check, then run the research suites and verify them.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
L=/mnt/quant_storage/quant_workspace/logs
P=/opt/qenv/bin/python
export LEAN_BIN=/opt/qenv/bin/lean
export PATH=/opt/qenv/bin:$PATH
log() { echo "$(date -u '+%F %T') $*" >> $L/pipeline.log; }
fail() { log "PIPELINE_FAILED: $*"; touch $L/PIPELINE_FAILED; exit 1; }

log "waiting for raw data and image"
until [ -f $L/XFER_DONE ]; do sleep 30; done
until grep -q PULL_DONE $L/docker_pull.log 2>/dev/null; do
  grep -q PULL_FAILED $L/docker_pull.log 2>/dev/null && fail "image pull failed"
  sleep 30
done
log "building LEAN data"
bash $W/etl/build_a409.sh > $L/build_a409.out 2>&1 || fail "data build (see build_a409.out)"
grep -q BUILD_DONE $L/build_a409.out || fail "data build did not finish"
grep -l -i "traceback\|error" $L/build_*_20*.out $L/build_*_underlying.out 2>/dev/null && fail "errors in build logs"
log "build done: $(du -sh $W/data | cut -f1)"

cd $W
log "environment check: BTC weekly SHORT_STRANGLE must reproduce the workstation result -8.175 %"
$P run_suite.py suites/env_check.json results/env_check.json --workers 1 >> $L/pipeline.log 2>&1
$P - >> $L/pipeline.log 2>&1 <<'EOF' || fail "environment check"
import json, sys
r = json.load(open("results/env_check.json"))[0]
ok = r.get("ok") and abs(r["metrics"]["total_return"] - (-0.0817516369)) < 0.0001
print("ENV_CHECK", "OK" if ok else "MISMATCH", r.get("metrics", {}).get("total_return"), r.get("error", ""))
sys.exit(0 if ok else 1)
EOF
log "running research suites"
$P run_suite.py suites/volsell_finalists.json results/volsell_finalists.json --workers 12 >> $L/pipeline.log 2>&1 &
$P run_suite.py suites/intraday_full.json results/intraday_full.json --workers 14 >> $L/pipeline.log 2>&1 &
wait
for s in volsell_finalists intraday_full; do
  $P verify_v2.py results/$s.json results/verify_$s.json > results/verify_$s.txt 2>&1
done
log "PIPELINE_DONE"
touch $L/PIPELINE_DONE
