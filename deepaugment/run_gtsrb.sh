set -x
run_gtsrb ()
{
    model=$1
    shift
    interval=30
    for i in $(seq 0 $interval 29)
    do  
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_gtsrt_${model}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -t $i -e $interval  -o -f 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model -o -f 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_gtsrt_${model}_$1_f.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -t $i -e $interval -f 2>&1 | tee -a execution_gtsrt_${model}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model -f 2>&1 | tee -a execution_gtsrt_${model}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_gtsrt_${model}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -t $i -e $interval -o 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model -o 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize.out
        else
            rm execution_gtsrt_${model}_$1.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -t $i -e $interval 2>&1 | tee -a execution_gtsrt_${model}_$1.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model 2>&1 | tee -a execution_gtsrt_${model}_$1.out
        fi
    done
}

run()
{
    #run_gtsrb $1 original
    #run_gtsrb $1 replace30
    run_gtsrb $1 replace_worst_of_10
    run_gtsrb $1 ga_loss

    #run_gtsrb $1 replace_worst_of_10 -o
    #run_gtsrb $1 ga_loss -o

    #run_gtsrb $1 original -f
    #run_gtsrb $1 replace30 -f
    #run_gtsrb $1 replace_worst_of_10 -f
    #run_gtsrb $1 ga_loss -f

    #run_gtsrb $1 replace_worst_of_10 -f -o
    #run_gtsrb $1 ga_loss -f -o
}

#run 0
#run 1
#run 2
run 3
