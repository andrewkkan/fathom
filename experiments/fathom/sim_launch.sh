# Copyright 2022 FATHOM Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
ROOT_DIR='.'
FLAGS_DIR=$ROOT_DIR'/experiments/fathom/'
EXP_DIRs=`ls -d $FLAGS_DIR/*_exp_config/`
OUTPUT_DIR=$FLAGS_DIR

SEEDs=`seq 1 50`
EXP_REGEX='\/([a-zA-Z0-9\-\_]+)\_exp\_config'
CONFIG_REGEX='\/([a-zA-Z0-9\-\_]+)\.flags'

for EXP_DIR in $EXP_DIRs
do
    [[ $EXP_DIR =~ $EXP_REGEX ]]
    EXP_NAME="${BASH_REMATCH[1]}"
    RESULTS_DIR=$OUTPUT_DIR"/results_"$EXP_NAME
    mkdir -p $RESULTS_DIR
    # echo $RESULTS_DIR
    EXP_CONFIGs=`ls $EXP_DIR/*.flags`
    for EXP_CONFIG in $EXP_CONFIGs
    do
        [[ $EXP_CONFIG =~ $CONFIG_REGEX ]]
        CONFIG_NAME="${BASH_REMATCH[1]}"
        for seed in $SEEDs
        do
            SIM_CMD='python3 experiments/fathom/run_fathom.py --flagfile='$EXP_CONFIG' --sim_seed='$seed' --logfile '$RESULTS_DIR'/'$CONFIG_NAME'_seed'$seed
            $SIM_CMD
        done
    done
done
