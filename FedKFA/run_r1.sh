#!/bin/sh
ROUND=1
CLIENTS=10
EPOCHS=10
for i in 0.01
do
    python experiments.py --model=mlp \
        --dataset=fmnist \
        --alg=fedavg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=$EPOCHS \
        --n_parties=$CLIENTS \
        --rho=0.9 \
        --comm_round=$ROUND \
        --partition=noniid-labeldir \
        --beta=$i\
        --device='cuda:0'\
        --datadir='./data/' \
        --logdir='./logs/' \
        --noise=0\
        --init_seed=0 &
    sleep 1
    python experiments.py --model=mlp \
        --dataset=fmnist \
        --alg=fedprox \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=$EPOCHS \
        --n_parties=$CLIENTS \
        --rho=0.9 \
        --comm_round=$ROUND \
        --partition=noniid-labeldir \
        --beta=$i\
        --device='cuda:0'\
        --datadir='./data/' \
        --logdir='./logs/' \
        --noise=0\
        --init_seed=0 &
    sleep 1
    python experiments.py --model=mlp \
        --dataset=fmnist \
        --alg=scaffold \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=$EPOCHS \
        --n_parties=$CLIENTS \
        --rho=0.9 \
        --comm_round=$ROUND \
        --partition=noniid-labeldir \
        --beta=$i\
        --device='cuda:0'\
        --datadir='./data/' \
        --logdir='./logs/' \
        --noise=0\
        --init_seed=0 &
    sleep 1
    python experiments.py --model=mlp \
        --dataset=fmnist \
        --alg=fednova \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=$EPOCHS \
        --n_parties=$CLIENTS \
        --rho=0.9 \
        --comm_round=$ROUND \
        --partition=noniid-labeldir \
        --beta=$i\
        --device='cuda:0'\
        --datadir='./data/' \
        --logdir='./logs/' \
        --noise=0\
        --init_seed=0 &
    wait
    echo "For $i"
done