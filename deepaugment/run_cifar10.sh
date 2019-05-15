set -x
run_cifar10 ()
{
    rm /tmp/config*
    model=$1
    shift
    queue=$1
    shift
    interval=200
    for i in $(seq 0 $interval 199)
    do
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_cifar10_${model}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -q $queue -t $i -e $interval  -o -f 2>&1 | tee -a execution_cifar10_${model}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model -o -f 2>&1 | tee -a execution_cifar10_${model}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_cifar10_${model}_$1_f.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -q $queue -t $i -e $interval -f 2>&1 | tee -a execution_cifar10_${model}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model -f 2>&1 | tee -a execution_cifar10_${model}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_cifar10_${model}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -q $queue -t $i -e $interval -o 2>&1 | tee -a execution_cifar10_${model}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model -o 2>&1 | tee -a execution_cifar10_${model}_$1_optimize.out
        else
            rm execution_cifar10_${model}_$1_q_$queue.out
            python augmented_training.py --strategy $1 --dataset cifar10 -m $model -q $queue -t $i -e $interval 2>&1 | tee -a execution_cifar10_${model}_$1_q${queue}.out
            python adversarial_attack.py --strategy $1 --dataset cifar10 -m $model 2>&1 | tee -a execution_cifar10_${model}_$1_q${queue}.out
        fi
    done
}

run()
{
    #run_cifar10 $1 10 original
    #run_cifar10 $1 10 replace30
    #run_cifar10 $1 10 replace_worst_of_10
    run_cifar10 $1 10 ga_loss
    #run_cifar10 $1 10 ga_cov

    #run_cifar10 $1 10 replace_worst_of_10 -o
    #run_cifar10 $1 10 ga_loss -o

    #run_cifar10 $1 10 original -f
    #run_cifar10 $1 10 replace30 -f
    #run_cifar10 $1 10 replace_worst_of_10 -f
    #run_cifar10 $1 10 ga_loss -f

    #run_cifar10 $1 10 replace_worst_of_10 -f -o
    #run_cifar10 $1 10 ga_loss -f -o
}

#run 1
#run 3
#run 0
#run 2 #wide-resnet

#run_cifar10 1 3 ga_loss
#run_cifar10 1 5 ga_loss
#run_cifar10 1 10 ga_loss
#run_cifar10 1 15 ga_loss
#run_cifar10 1 20 ga_loss
#run_cifar10 1 30 ga_loss
#run_cifar10 1 50 ga_loss

for i in 1 5
do
    queue=10
    model=1
    dataset=cifar10
    rm /tmp/config*
    rm execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
    python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -r $i -q $queue -t 0 -e 200 -o 2>&1 | tee -a execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
    python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model -o 2>&1 | tee -a execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
done

