#!/bin/zsh
# ablation run: ./abl.sh TAG ARM SEED, any knob from env (LAYERS, WIDTH, NF, NV, FACTZIPF, STEPS, ...); appends to ablation_runs.jsonl
# ARM: base = normal attention + value embedding; slots = gate + 4 context-chosen slots, no W_v, every layer;
#      keepv = gate + 4 slots in layers after the first, W_v kept
L=${LAYERS:-2}; ALL=$(python3 -c "print(''.join(map(str,range($L))))"); LATER=${ALL:1}
case $2 in
  slots) export VE_GATE=1 VE_SLOTS=4 FREEZE=v V_ZERO=1 VS_L=$ALL V_ZERO_L=$ALL ;;
  keepv) export VE_GATE=1 VE_SLOTS=4 VS_L=$LATER ;;
esac
./suite_run.sh $3 > /dev/null
tail -1 sampled_runs.jsonl | python3 -c "import json,sys,os; r=json.loads(sys.stdin.read()); print(json.dumps({'tag':'$1','arm':'$2','seed':$3,'layers':$L,'width':int(os.environ.get('WIDTH',64)),'last':r['last']}))" >> ablation_runs.jsonl
