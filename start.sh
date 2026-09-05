# Codex
source /data/home/scwb792/clashctl/scripts/cmd/clashctl.sh
clashctl on
export PATH="/data/home/scwb792/run/lyt/npm-global/bin:$PATH"
unset ALL_PROXY all_proxy
export CODEX_HOME="/data/home/scwb792/run/mwj/codex-home"
codex --dangerously-bypass-approvals-and-sandbox

git -c http.proxy=http://127.0.0.1:7891 -c https.proxy=http://127.0.0.1:7891 push