#!/bin/bash
#
#SBATCH --job-name=tone_mapping
#SBATCH --output=res_tonemapping_mpi.md
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1

#TMO="gamma"
#
#echo "imagen|tiempo|id worker"
#echo ":---:|:---:|:---:"
#
#mpirun build/tonemapping ../images results/results_$TMO \| $TMO 1.2 0.4 #gamma
#mpirun build/tonemapping ../images results/results_$TMO \| $TMO 1 1 #log
#mpirun build/tonemapping ../images results/results_$TMO \| $TMO 1 150 #adap_log

for tmo in "gamma" "log" "adap_log"
do
	echo "**Operador: $tmo**"
	echo
	echo "imagen|tiempo|id worker"
	echo ":---:|:---:|:---:"
	if [ "${tmo}" == "gamma" ]; then
#		echo "${tmo}"
		mpirun build/tonemapping ../images results/results_$tmo \| $tmo 1.2 0.4 #gamma
	elif [ "${tmo}" == "log" ]; then
#	    echo "${tmo}"
		mpirun build/tonemapping ../images results/results_$tmo \| $tmo 1 1 #log
	elif [ "${tmo}" == "adap_log" ]; then
#		echo "${tmo}"
		mpirun build/tonemapping ../images results/results_$tmo \| $tmo 1 150 #adap_log
	fi
	echo
done
