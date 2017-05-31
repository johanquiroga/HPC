#!/bin/bash
#
#SBATCH --job-name=tone_mapping
#SBATCH --output=res_tonemapping_parallel.md
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#TMO="adap_log"

#echo "imagen|tiempo"
#echo ":---:|:---:"

#srun tonemap ../images results/results_$TMO \| $TMO 1.2 0.4 #gamma
#srun tonemap ../images results/results_$TMO \| $TMO 1 1 #log
#srun tonemap ../images results/results_$TMO \| $TMO 1 150 #adap_log

for tmo in "gamma" "log" "adap_log"
do
	echo "**Operador: $tmo**"
	echo
	echo "imagen|tiempo"
	echo ":---:|:---:"
	if [ "${tmo}" == "gamma" ]; then
#		echo "${tmo}"
	    srun tonemap ../images results/results_$tmo \| $tmo 1.2 0.4 #gamma
	elif [ "${tmo}" == "log" ]; then
#	    echo "${tmo}"
	    srun tonemap ../images results/results_$tmo \| $tmo 1 1 #log
	elif [ "${tmo}" == "adap_log" ]; then
#		echo "${tmo}"
		srun tonemap ../images results/results_$tmo \| $tmo 1 150 #adap_log
	fi
	echo
done