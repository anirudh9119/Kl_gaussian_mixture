#!/usr/bin/env python

from Distribution import Gaussian1D
from MM import GMM
import time

def __benchmark( mm1, mm2, repeat=1000 ):
    for sample_size in [ 1e3, 1e4, 1e5, 1e6 ]:
        start_t = time.time()
        for i in range( repeat ):
            mm1.kl_sampling( mm2, sample_size )
            mm2.kl_sampling( mm1, sample_size )
        speed = ( time.time() - start_t ) * 1000. / (repeat*2)
        print('sample_size=%8d speed=%6.0fms/run' % ( sample_size, speed ) )

    start_t = time.time()
    for i in range( repeat ):
        mm1.kl_lse_bound( mm2 )
        mm2.kl_lse_bound( mm1 )
    speed = ( time.time() - start_t ) * 1000. / (repeat*2)
    print('LSE Bounds           speed=%6.0fms/run' % speed )

def benchmark():
    print( 'GMM' )
    gaussians1 = []
    gaussians2 = []
    for i in range( 10 ):
        gaussians1.append( Gaussian1D( i, 1 )    )
        gaussians2.append( Gaussian1D( i+.5, 1 ) )
    gmm1 = GMM( gaussians1 )
    gmm2 = GMM( gaussians2 )
    __benchmark( gmm1, gmm2, repeat=1 )

if __name__ == '__main__':
    benchmark()
