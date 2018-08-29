#!/bin/sh

# estimating entropy of Gaussian Mixtures
echo '======= entropy experiments ========'
./test.py --entropy testcases/gmm1
./test.py --entropy testcases/gmm2
./test.py --entropy testcases/gmm3
./test.py --entropy testcases/gmm4
./test.py --entropy testcases/gmm5
./test.py --entropy testcases/gmm6

