### learning on long-tailed CIFAR10 ###
#ratio=100
python3 train.py -d CIFAR10 -g 0 -r 0.01
#ratio=200
python3 train.py -d CIFAR10 -g 0 -r 0.005
#ratio=500
python3 train.py -d CIFAR10 -g 0 -r 0.002

### learning on long-tailed CIFAR100 ###
#ratio=100
python3 train.py -d CIFAR100 -g 0 -r 0.01
#ratio=200
python3 train.py -d CIFAR100 -g 0 -r 0.005
#ratio=500
python3 train.py -d CIFAR100 -g 0 -r 0.002


### learning with LDAM loss ###
python3 train.py -d CIFAR10 -g 0 -r 0.01 -l LDAM

### learning with LogitAdjust ###
python3 train.py -d CIFAR10 -g 0 -r 0.01 -l LogitAdjust

### learning with FCE loss type alpha ###
python3 train.py -d CIFAR10 -g 0 -r 0.01 -l FCE_a

### learning with FCE loss type beta ###
python3 train.py -d CIFAR10 -g 0 -r 0.01 -l FCE_b
