set -x

dataset=svhn
epoch=50
optimize=false
operator=3

run_all()
{
    for model in 0 # 1
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

