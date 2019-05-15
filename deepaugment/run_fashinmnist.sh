set -x

dataset=fashionmnist
epoch=50
optimize=false
operator=3

run_all()
{
    for model in 0 # 1 2 
    do
        for approach in replace_worst_of_10 # original replace30 replace_worst_of_10 ga_loss ga_cov 
        do
            log_file=execution_${dataset}_${model}_${approach}
            if $optimize 
            then
                flag=-o
                log_file=${log_file}_optimize
            elif [ $operator == 6 ]
            then 
                flag=-f
                log_file=${log_file}_operator6
            fi 
            log_file=${log_file}.out

            rm /tmp/config*
            rm $log_file
            python augmented_training.py --strategy $approach --dataset $dataset -m $model -t 0 -e ${epoch} $flag 2>&1 | tee -a $log_file
            python adversarial_attack.py --strategy $approach --dataset $dataset -m $model $flag 2>&1 | tee -a $log_file
        done
    done 
}

various_loss_threshold()
{
    # threshold 1e-1, 1e-2 ... 1e-5
    for i in 1 2 3 4 5
    do
        model=0
        flag=-o
        rm /tmp/config*
        log_file=execution_${dataset}_${model}_ga_loss_t_${i}_${flag}.out
        rm $log_file
        python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -r $i -t 0 -e ${epoch} $flag 2>&1 | tee -a $log_file
        python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model $flag 2>&1 | tee -a $log_file
    done
}

various_popsize()
{
    for queue in 3 5 10 15 20 30
    do
        model=0
        log_file=execution_${dataset}_${model}_ga_loss_q_${queue}.out
        rm /tmp/config*
        rm $log_file
        python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -q $queue -t 0 -e ${epoch} 2>&1 | tee -a $log_file
        python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model 2>&1 | tee -a $log_file
    done
}

run_all
