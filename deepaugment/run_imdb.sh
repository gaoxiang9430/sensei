set -x
run_imdb ()
{
    interval=50
    model_id=$1
    shift
    for i in $(seq 0 $interval 19)
    do
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_imdb_${model_id}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset imdb -m $model_id -t $i -e $interval -o -f 2>&1 | tee -a execution_imdb_${model_id}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset imdb -m $model_id -o -f 2>&1 | tee -a execution_imdb_${model_id}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_imdb_${model_id}_$1_f.out
            python augmented_training.py --strategy $1 --dataset imdb -m $model_id -t $i -e $interval -f 2>&1 | tee -a execution_imdb_${model_id}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset imdb -m $model_id -f 2>&1 | tee -a execution_imdb_${model_id}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_imdb_${model_id}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset imdb -m $model_id -t $i -e $interval -o 2>&1 | tee -a execution_imdb_${model_id}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset imdb -m $model_id -o 2>&1 | tee -a execution_imdb_${model_id}_$1_optimize.out
        else
            rm execution_imdb_${model_id}_$1.out
            python augmented_training.py --strategy $1 --dataset imdb -m $model_id -t $i -e $interval 2>&1 | tee -a execution_imdb_${model_id}_$1.out
            python adversarial_attack.py --strategy $1 --dataset imdb -m $model_id 2>&1 | tee -a execution_imdb_${model_id}_$1.out
        fi
    done
}

run_all()
{
    model_id=$1
    #run_imdb ${model_id} original
    #run_imdb $model_id replace30
    run_imdb $model_id replace_worst_of_10
    #run_imdb $model_id ga_loss

    #run_imdb $model_id replace_worst_of_10 -o
    #run_imdb $model_id ga_loss -o

    #run_imdb $model_id original -f
    #run_imdb $model_id replace30 -f
    #run_imdb $model_id replace_worst_of_10 -f
    #run_imdb $model_id ga_loss -f

    #run_imdb $model_id replace_worst_of_10 -f -o
    #run_imdb $model_id ga_loss -f -o
}

#run_imdb 0 ga_loss
#run_all 0
run_all 1
