set -x
run_gtsrb ()
{
    interval=30
    for i in $(seq 0 $interval 29)
    do  
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            python augmented_training.py --strategy $1 --dataset gtsrb -t $i -e $interval  -o -f 2>&1 | tee -a execution_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -o -f 2>&1 | tee -a execution_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            python augmented_training.py --strategy $1 --dataset gtsrb -t $i -e $interval -f 2>&1 | tee -a execution_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -f 2>&1 | tee -a execution_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            python augmented_training.py --strategy $1 --dataset gtsrb -t $i -e $interval -o 2>&1 | tee -a execution_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -o 2>&1 | tee -a execution_$1_optimize.out
        else
            python augmented_training.py --strategy $1 --dataset gtsrb -t $i -e $interval 2>&1 | tee -a execution_$1.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb 2>&1 | tee -a execution_$1.out
        fi
    done
}

#run_gtsrb original
#run_gtsrb replace30
#run_gtsrb replace40
#run_gtsrb replace_worst_of_10
#run_gtsrb ga_loss

#run_gtsrb replace_worst_of_10 -o
#run_gtsrb ga_loss -o

#run_gtsrb original -f
#run_gtsrb replace30 -f
run_gtsrb replace_worst_of_10 -f
#run_gtsrb ga_loss -f

#run_gtsrb replace_worst_of_10 -f -o
#run_gtsrb ga_loss -f -o

