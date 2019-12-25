#/bin/bash

set -x 

dataset=gtsrb
epoch=30
model="1"

sa=false
operator=3

run_all()
{
    for m in $model
    do
        for approach in original replace30 replace_worst_of_10 ga_loss #ga_cov
        do
            log_file=execution_${dataset}_${m}_${approach}
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
            python augmented_training.py -m $m -t 0 -e ${epoch} $flag $approach $dataset 2>&1 | tee -a $log_file
            python adversarial_attack.py -m $m $flag $approach $dataset 2>&1 | tee -a $log_file
        done
    done
}

various_loss_threshold()
{
    # threshold 1e-1, 1e-2 ... 1e-5
    for m in $model
    do
        for i in 1 2 3 4 5
        do
            m=3
            flag=-o
            rm -rf /tmp/config*
            log_file=execution_${dataset}_${m}_ga_loss_t_${i}_${flag}.out
            rm -rf $log_file
            python augmented_training.py ga_loss $dataset -m $m -r $i -t 0 -e ${epoch} $flag 2>&1 | tee -a $log_file
            python adversarial_attack.py ga_loss $dataset -m $m $flag 2>&1 | tee -a $log_file
        done
    done
}

various_popsize()
{
    for m in $model
    do
        for queue in 3 5 10 15 20 30
        do
            m=2
            log_file=execution_${dataset}_${m}_ga_loss_q_${queue}.out
            rm -rf /tmp/config*
            rm -rf $log_file
            python augmented_training.py ga_loss $dataset -m $m -q $queue -t 0 -e ${epoch} 2>&1 | tee -a $log_file
            python adversarial_attack.py ga_loss $dataset -m $m 2>&1 | tee -a $log_file
        done
    done
}

config()
{
    dataset=$1
    if [ $dataset = gtsrb ]
    then
        epoch=30
        model="0 1 2 3"
    elif [ $dataset = cifar10 ]
    then
        epoch=200
        model="0 1 2 3"
    elif [ $dataset = fashionmnist ]
    then
        epoch=50
        model="0 1 2"
    elif [ $dataset = imdb ]
    then
        epoch=50
        model="0 1"
    elif [ $dataset = svhn ]
    then
        epoch=50
        model="0 1"
    else
        echo "unsupported dataset $dataset!!!"
        exit 2
    fi
}

if [ "$#" -lt 1 ]
then
    echo -e "USAGE: ./run.sh DATASET [options]\n"
    echo "  dataset: supported dataset: gtsrb, fashionmnist, cifar10, imdb, svhn"
    echo "  options: supported options: sa (sensei_sa), st (six transformation), ps (various popsize), lt (different loss threshold)]!"
    exit 1
fi

op=none
if [ "$#" -ge 1 ]
then
    dataset=$1
    shift
fi
if [ "$#" -ge 1 ]
then
    op=$1
fi

# config
if [ $op = sa ]
then
    sa=true
elif [ $op = st ]
then
    operator=6
fi
config $dataset

if [[ "$op" == "ps" ]];
then
    various_popsize
elif [[ "$op" == "lt" ]];
then
    various_loss_threshold
else
    run_all
fi

