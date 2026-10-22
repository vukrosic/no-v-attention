#!/bin/zsh
# realistic 5 s suite: small vocab, no MoE, 400 steps, 8 base skills + 5 multi-step skills sized to be learnable
export NV=256 MOE=0 STEPS=${STEPS:-400} EXTRA=${EXTRA:-rfact,agree,rrecall,bind,sum3} EXTRA_W=${EXTRA_W:-0.5} RN=${RN:-2} BA=${BA:-2} SG=${SG:-1} SD=${SD:-2} SM=${SM:-5}
./toy.sh $1
