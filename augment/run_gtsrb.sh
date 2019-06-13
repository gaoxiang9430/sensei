set -x

dataset=gtsrb
epoch=30
sa=false
operator=3

run_all()
{
    for model in 0 # 1 2 3
    do
        for approach in replace_worst_of_10 # original replace30 replace_worst_of_10 ga_loss ga_cov
        do
            log_file=execution_${dataset}_${model}_${approach}
            if $sa 
            then
                flag=-o
                log_file=${log_file}_sa
            elif [ $operator == 6 ]
            then
                flag=-f
                log_file=${log_file}_operator6
            fi
            log_file=${log_file}.out

            rm -rf /tmp/config*
            rm -rf $log_file
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
        model=3
        flag=-o
        rm -rf /tmp/config*
        log_file=execution_${dataset}_${model}_ga_loss_t_${i}_${flag}.out
        rm -rf $log_file
        python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -r $i -t 0 -e ${epoch} $flag 2>&1 | tee -a $log_file
        python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model $flag 2>&1 | tee -a $log_file
    done
}

various_popsize()
{
    for queue in 3 5 10 15 20 30
    do
        model=2
        log_file=execution_${dataset}_${model}_ga_loss_q_${queue}.out
        rm -rf /tmp/config*
        rm -rf $log_file
        python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -q $queue -t 0 -e ${epoch} 2>&1 | tee -a $log_file
        python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model 2>&1 | tee -a $log_file
    done
}

run_all
