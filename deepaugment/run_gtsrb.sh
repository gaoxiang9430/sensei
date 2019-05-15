set -x
run_gtsrb ()
{
    rm /tmp/config*
    model=$1
    shift
    queue=$1
    shift
    interval=30
    for i in $(seq 0 $interval 29)
    do  
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_gtsrt_${model}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -q $queue -t $i -e $interval  -o -f 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model -o -f 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_gtsrt_${model}_$1_f.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -q $queue -t $i -e $interval -f 2>&1 | tee -a execution_gtsrt_${model}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model -f 2>&1 | tee -a execution_gtsrt_${model}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_gtsrt_${model}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -q $queue -t $i -e $interval -o 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model -o 2>&1 | tee -a execution_gtsrt_${model}_$1_optimize.out
        else
            rm execution_gtsrt_${model}_$1.out
            python augmented_training.py --strategy $1 --dataset gtsrb -m $model -q $queue -t $i -e $interval 2>&1 | tee -a execution_gtsrt_${model}_$1_q${queue}.out
            python adversarial_attack.py --strategy $1 --dataset gtsrb -m $model 2>&1 | tee -a execution_gtsrt_${model}_$1_q${queue}.out
        fi
    done
}

run()
{
    #run_gtsrb $1 10 original
    #run_gtsrb $1 10 replace30
    #run_gtsrb $1 10 replace_worst_of_10
    run_gtsrb $1 10 ga_loss
    #run_gtsrb $1 10 ga_cov

    #run_gtsrb $1 10 replace_worst_of_10 -o
    #run_gtsrb $1 10 ga_loss -o

    #run_gtsrb $1 10 original -f
    #run_gtsrb $1 10 replace30 -f
    #run_gtsrb $1 10 replace_worst_of_10 -f
    #run_gtsrb $1 10 ga_loss -f

    #run_gtsrb $1 10 replace_worst_of_10 -f -o
    #run_gtsrb $1 10 ga_loss -f -o
}

run_all()
{
    run 0
    run 1
    run 2
    run 3
}

run_popsize()
{
    #run_gtsrb 2 3 ga_loss
    #run_gtsrb 2 5 ga_loss
    #run_gtsrb 2 10 ga_loss
    #run_gtsrb 2 15 ga_loss
    #run_gtsrb 2 20 ga_loss
    run_gtsrb 2 30 ga_loss
    run_gtsrb 2 50 ga_loss
    #for i in {3,5,10,15,20,30,50}
    #do
    #    run_gtsrb 2 $i ga_loss 
    #done
}

for i in 1 2 3 4 5
do
    queue=10
    model=3
    dataset=gtsrb
    rm /tmp/config*
    rm execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
    python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -r $i -q $queue -t 0 -e 30 -o 2>&1 | tee -a execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
    python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model -o 2>&1 | tee -a execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
done
