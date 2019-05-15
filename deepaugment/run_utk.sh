set -x
run_utk ()
{
    rm /tmp/config*
    interval=50
    model_id=$1
    shift
    for i in $(seq 0 $interval 19)
    do
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_utk_${model_id}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset utk -m $model_id -t $i -e $interval -o -f 2>&1 | tee -a execution_utk_${model_id}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset utk -m $model_id -o -f 2>&1 | tee -a execution_utk_${model_id}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_utk_${model_id}_$1_f.out
            python augmented_training.py --strategy $1 --dataset utk -m $model_id -t $i -e $interval -f 2>&1 | tee -a execution_utk_${model_id}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset utk -m $model_id -f 2>&1 | tee -a execution_utk_${model_id}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_utk_${model_id}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset utk -m $model_id -t $i -e $interval -o 2>&1 | tee -a execution_utk_${model_id}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset utk -m $model_id -o 2>&1 | tee -a execution_utk_${model_id}_$1_optimize.out
        else
            rm execution_utk_${model_id}_$1.out
            python augmented_training.py --strategy $1 --dataset utk -m $model_id -t $i -e $interval 2>&1 | tee -a execution_utk_${model_id}_$1.out
            python adversarial_attack.py --strategy $1 --dataset utk -m $model_id 2>&1 | tee -a execution_utk_${model_id}_$1.out
        fi
    done
}

run_all()
{
    model_id=$1
    #run_utk ${model_id} original
    #run_utk $model_id replace30
    #run_utk $model_id replace_worst_of_10
    run_utk $model_id ga_loss
    #run_utk $model_id ga_cov

    #run_utk $model_id replace_worst_of_10 -o
    #run_utk $model_id ga_loss -o

    #run_utk $model_id original -f
    #run_utk $model_id replace30 -f
    #run_utk $model_id replace_worst_of_10 -f
    #run_utk $model_id ga_loss -f

    #run_utk $model_id replace_worst_of_10 -f -o
    #run_utk $model_id ga_loss -f -o
}

run_all 0
run_all 1
