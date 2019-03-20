set -x
run_cifar10 ()
{
    model=$1
    shift
    interval=200
    for i in $(seq 0 $interval 199)
    do
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_cifar10_${model}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -t $i -e $interval  -o -f 2>&1 | tee -a execution_cifar10_${model}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model -o -f 2>&1 | tee -a execution_cifar10_${model}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_cifar10_${model}_$1_f.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -t $i -e $interval -f 2>&1 | tee -a execution_cifar10_${model}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model -f 2>&1 | tee -a execution_cifar10_${model}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_cifar10_${model}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -t $i -e $interval -o 2>&1 | tee -a execution_cifar10_${model}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model -o 2>&1 | tee -a execution_cifar10_${model}_$1_optimize.out
        else
            rm execution_cifar10_${model}_$1.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -t $i -e $interval 2>&1 | tee -a execution_cifar10_${model}_$1.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model 2>&1 | tee -a execution_cifar10_${model}_$1.out
        fi
    done
}

run()
{
    #run_cifar10 $1 original
    #run_cifar10 $1 replace30
    #run_cifar10 $1 replace_worst_of_10
    run_cifar10 $1 ga_loss

    #run_cifar10 $1 replace_worst_of_10 -o
    #run_cifar10 $1 ga_loss -o

    #run_cifar10 $1 original -f
    #run_cifar10 $1 replace30 -f
    #run_cifar10 $1 replace_worst_of_10 -f
    #run_cifar10 $1 ga_loss -f

    #run_cifar10 $1 replace_worst_of_10 -f -o
    #run_cifar10 $1 ga_loss -f -o
}

#run 1
#run 2 #vgg
run 3
run 5
run 4
#run 0
