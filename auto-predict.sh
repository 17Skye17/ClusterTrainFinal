#!/bin/bash
dir=$(ls -l ./ClusterTrain-4-26-g2305-4/ |awk '/^d/{print $NF}')
file=predict.sh
evaluation=evaluator.py
result=RMSE_ClusterTrain-4-26-g2305-4.txt
for i in $dir
do
	    VAR=`sed -n '4p' $file`
		IFS='/' arr=($VAR)
		NEWLINE=${arr[0]}/$i
        sed -i "4c ${NEWLINE}" $file

		#run predict
        sh $file
		
		#run evaluator
		python $evaluation >> $result
done
