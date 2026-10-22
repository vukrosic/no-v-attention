#!/bin/zsh
# ablation run: ./ablation.sh TAG ARM SEED, any knob from env (LAYERS, WIDTH, NF, NV, FACTZIPF, STEPS, ...); appends to ablation_runs.jsonl
# ARM: base = normal attention + value embedding; slots = gate + 4 context-chosen slots, no W_v, every layer;
#      keepv = gate + 4 slots in layers after the first, W_v kept
#      novXY = gate + 4 slots in every layer, W_v removed only in layers X,Y (nov = W_v kept everywhere)
#      splitXY = layers X,Y use only value-embedding slots, the other layers use only W_v
L=${LAYERS:-2}; ALL=$(python3 -c "print(''.join(map(str,range($L))))"); LATER=${ALL:1}
case $2 in
  slots) export VE_GATE=1 VE_SLOTS=4 FREEZE=v V_ZERO=1 VS_L=$ALL V_ZERO_L=$ALL ;;
  keepv) export VE_GATE=1 VE_SLOTS=4 VS_L=$LATER ;;
  nov*) NOV=${2#nov}; export VE_GATE=1 VE_SLOTS=4 VS_L=$ALL V_ZERO=1 V_ZERO_L=${NOV:-x}
        export FREEZE=$(python3 -c "print(','.join('v'+c for c in '$NOV'))") ;;   # novXY: slots every layer, W_v removed only in layers X,Y
  split*) S=${2#split}; export VE_GATE=1 VE_SLOTS=4 VS_L=$S V_ZERO=1 V_ZERO_L=$S VE_LAYERS=$(python3 -c "print(','.join('$S'))")
        export FREEZE=$(python3 -c "print(','.join('v'+c for c in '$S'))") ;;   # splitXY: layers X,Y = slots only, no W_v; other layers = W_v only, no value embedding
esac
./suite_run.sh $3 > /dev/null
tail -1 sampled_runs.jsonl | python3 -c "import json,sys,os; r=json.loads(sys.stdin.read()); print(json.dumps({'tag':'$1','arm':'$2','seed':$3,'layers':$L,'width':int(os.environ.get('WIDTH',64)),'last':r['last']}))" >> ablation_runs.jsonl
