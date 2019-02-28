set -x
run_gtsrb ()
{
    rm -rf execution_$1.out
    interval=30
    for i in $(seq 0 $interval 29)
    do
        python augmented_training.py --strategy $1 --dataset gtsrb -t $i -e $interval &>> execution_$1.out
    done
    python adversarial_attack.py --strategy $1 --dataset gtsrb &>> execution_$1.out
}

#run_gtsrb original
#run_gtsrb replace30
#run_gtsrb replace40
#run_gtsrb replace_worst_of_10_cov
#run_gtsrb ga_loss

run_cifar ()
{
    rm -rf execution_cifar_$1.out
    interval=50
    for i in $(seq 0 $interval 199)
    do
        python augmented_training.py --strategy $1 --dataset cifar10 -t $i -e $interval &>> execution_cifar_$1.out
    done
    python adversarial_attack.py --strategy $1 --dataset cifar10 &>> execution_cifar_$1.out
}

run_cifar original
run_cifar replace30
run_cifar replace40
