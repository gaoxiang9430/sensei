set -x
run_fashionmnist ()
{
    rm /tmp/config*
    interval=100
    model_id=$1
    shift
    queue=$1
    shift
    for i in $(seq 0 $interval 19)
    do
        if [ "$#" -gt 2 ] && [ "$2" == "-f" ] && [ "$3" == "-o" ]
        then
            rm execution_fashionmnist_${model_id}_$1_optimize_f.out
            python augmented_training.py --strategy $1 --dataset fashionmnist -m $model_id -q $queue -t $i -e $interval -o -f 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_optimize_f.out
            python adversarial_attack.py --strategy $1 --dataset fashionmnist -m $model_id -o -f 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_optimize_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-f" ]
        then
            rm execution_fashionmnist_${model_id}_$1_f.out
            python augmented_training.py --strategy $1 --dataset fashionmnist -m $model_id -q $queue -t $i -e $interval -f 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_f.out
            python adversarial_attack.py --strategy $1 --dataset fashionmnist -m $model_id -f 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_f.out
        elif [ "$#" -gt 1 ] && [ "$2" == "-o" ]
        then
            rm execution_fashionmnist_${model_id}_$1_optimize.out
            python augmented_training.py --strategy $1 --dataset fashionmnist -m $model_id -q $queue -t $i -e $interval -o 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_optimize.out
            python adversarial_attack.py --strategy $1 --dataset fashionmnist -m $model_id -o 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_optimize.out
        else
            rm execution_fashionmnist_${model_id}_$1.out
            python augmented_training.py --strategy $1 --dataset fashionmnist -m $model_id -q $queue -t $i -e $interval 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_q${queue}.out
            python adversarial_attack.py --strategy $1 --dataset fashionmnist -m $model_id 2>&1 | tee -a execution_fashionmnist_${model_id}_$1_q${queue}.out
        fi
    done
}

run_all()
{
    model_id=$1
    #run_fashionmnist ${model_id} 10 $original
    #run_fashionmnist $model_id 10 $replace30
    #run_fashionmnist $model_id 10 $replace_worst_of_10
    run_fashionmnist $model_id 10 $ga_loss
    #run_fashionmnist $model_id 10 $ga_cov

    #run_fashionmnist $model_id 10 $replace_worst_of_10 -o
    #run_fashionmnist $model_id 10 $ga_loss -o

    #run_fashionmnist $model_id 10 $original -f
    #run_fashionmnist $model_id 10 $replace30 -f
    #run_fashionmnist $model_id 10 $replace_worst_of_10 -f
    #run_fashionmnist $model_id 10 $ga_loss -f

    #run_fashionmnist $model_id 10 $replace_worst_of_10 -f -o
    #run_fashionmnist $model_id 10 $ga_loss -f -o
}

#run_all 0
#run_all 1
#run_all 2 

#run_fashionmnist 0 3 ga_loss
#run_fashionmnist 0 5 ga_loss
#run_fashionmnist 0 10 ga_loss
#run_fashionmnist 0 15 ga_loss
#run_fashionmnist 0 20 ga_loss
#run_fashionmnist 0 30 ga_loss

for i in 1 5
do
    queue=10
    model=0
    dataset=fashionmnist
    rm /tmp/config*
    rm execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
    python augmented_training.py --strategy ga_loss --dataset $dataset -m $model -r $i -q $queue -t 0 -e 50 -o 2>&1 | tee -a execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
    python adversarial_attack.py --strategy ga_loss --dataset $dataset -m $model -o 2>&1 | tee -a execution_${dataset}_${model}_ga_loss_t_${i}_optimize.out
done

