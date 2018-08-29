#!/bin/sh

# estimating KL of Mixtures
echo '======= KL experiments ========'
./test.py --kl testcases/emm1 testcases/emm2
./test.py --kl testcases/gmm1 testcases/gmm2
./test.py --kl testcases/rmm1 testcases/rmm2
./test.py --kl testcases/gamm1 testcases/gamm2

