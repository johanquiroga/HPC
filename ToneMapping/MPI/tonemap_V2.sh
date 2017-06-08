#!/bin/bash

NODES=3

for tmo in "gamma" "log" "adap_log"
do
        echo "**Operador: $tmo**"
        echo
        echo "imagen|tiempo|id worker"
        echo ":---:|:---:|:---:"
        if [ "${tmo}" == "gamma" ]; then
#               echo "${tmo}"
                mpirun -np $NODES --hostfile hostfile build/tonemapping ../images results/results_$tmo \| $tmo 1.2 0.4 #gamma
        elif [ "${tmo}" == "log" ]; then
#           echo "${tmo}"
                mpirun -np $NODES --hostfile hostfile build/tonemapping ../images results/results_$tmo \| $tmo 1 1 #log
        elif [ "${tmo}" == "adap_log" ]; then
#               echo "${tmo}"
                mpirun -np $NODES --hostfile hostfile build/tonemapping ../images results/results_$tmo \| $tmo 1 100 #adap_log
        fi
        echo
done

