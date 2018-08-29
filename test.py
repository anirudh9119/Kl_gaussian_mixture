#!/usr/bin/env python

from __future__ import print_function

from MM import RMM, GMM, EMM, GaMM

import numpy as np
import argparse
import warnings
import sys, os, time

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc
rc( 'font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':14} )
rc( 'text', usetex=True )
#warnings.filterwarnings( "ignore", module="matplotlib" )

def load( name ):
    '''
    load mm from disk
    the filename should indicate the type of the mixture components
    '''
    if 'gmm' in name.lower():
        mm = GMM.load( name )

    elif 'rmm' in name.lower():
        mm = RMM.load( name )

    elif 'emm' in name.lower():
        mm = EMM.load( name )

    elif 'gamm' in name.lower():
        mm = GaMM.load( name )

    else:
        raise RuntimeError( 'cannot determin the mixture type' )

    return mm

def show_signal( ax, mm, title ):
    '''
    plot the mixture model
    '''
    x = np.linspace( mm.plot_left(), mm.plot_right(), 100 )
    y = np.exp( mm.loglikelihood( x ) )
    ax.plot( x, y )
    ax.set_xlim( mm.plot_left(), mm.plot_right() )
    ax.fill_between( x, 0, y )
    ax.set_title( title )

def show_entropy( ax, mm, title, repeat=100, init_seed=2016 ):
    sample_size_range = [ 10, 1e2, 1e3, 1e4 ]
    results = np.zeros( [ len(sample_size_range), repeat ] )

    for i, sample_size in enumerate( sample_size_range ):
        print( 'sample_size=%7d' % sample_size, end=" " )
        for j in range( repeat ):
            mm.seed( init_seed+j )
            results[i,j] = mm.entropy_sampling( sample_size )
        print( '%6.3f-%6.3f' % ( results[i,:].mean(), results[i,:].std() ) )

    bounds1 = mm.entropy_lse_bound( algo=1 )
    print( '          base bound [%.3f %.3f]' % bounds1 )

    bounds2 = mm.entropy_lse_bound( algo=2 )
    print( 'data-dependent bound [%.3f %.3f]' % bounds2, end=' ' )

    improve = 1-(bounds2[1]-bounds2[0])/(bounds1[1]-bounds1[0])
    improve *= 100
    print( '%.0f%% improvement' % improve )

    _min = results.min( 1 )
    _max = results.max( 1 )
    mean = results.mean( 1 )
    std  = results.std( 1 )

    x = [ i+0.5 for i,_ in enumerate(sample_size_range) ]
    ax.axhline( bounds1[0], linewidth=2, alpha=.6, color='blue' )
    ax.axhline( bounds1[1], linewidth=2, alpha=.6, color='red'  )

    ax.axhline( bounds2[0], linewidth=2, alpha=.6, color='green',  ls='dashed' )
    ax.axhline( bounds2[1], linewidth=2, alpha=.6, color='orange', ls='dashed' )

    ax.errorbar( x, mean, yerr=std, fmt='o', color='black', elinewidth=2 )

    try:
        upper_bound_gauss = mm.entropy_upper_bound()
        print( 'GMM upper bound %.3f' % upper_bound_gauss )
        ax.axhline( upper_bound_gauss, linewidth=2, alpha=.6, color='black', ls='dotted'  )
    except:
        pass

    blue   = mlines.Line2D( [], [], color='blue',   label='CELB' )
    red    = mlines.Line2D( [], [], color='red',    label='CEUB' )
    green  = mlines.Line2D( [], [], color='green',  label='CEALB', ls='dashed' )
    orange = mlines.Line2D( [], [], color='orange', label='CEAUB', ls='dashed' )
    black  = mlines.Line2D( [], [], color='black',  label='MEUB',  ls='dotted' )
    ax.legend( handles=[blue,red,green,orange,black], fontsize=10, loc='best', fancybox=True, framealpha=0.5 )

    ax.set_xticks( x )
    ax.set_xticklabels( [ r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$' ] )
    ax.set_xlim( 0, len(sample_size_range) )
    #ax.set_ylim( mean[0]-1.2*std[0], mean[0]+1.2*std[0] )
    ax.set_title( title + ( ' (%.0f\%%)' % improve ) )

def show_kl( ax, mm1, mm2, title, repeat=100, init_seed=2016 ):
    '''
    measure KL( mm1; mm2 )
    '''
    sample_size_range = [ 10, 1e2, 1e3, 1e4 ]
    results = np.zeros( [ len(sample_size_range), repeat ] )

    for i, sample_size in enumerate( sample_size_range ):
        print( 'sample_size=%7d' % sample_size, end=" " )
        for j in range( repeat ):
            mm1.seed( init_seed+j )
            mm2.seed( init_seed+j )
            results[i,j] = mm1.kl_sampling( mm2, sample_size )
        print( '%6.3f-%6.3f' % ( results[i,:].mean(), results[i,:].std() ) )

    bounds1 = mm1.kl_lse_bound( mm2, algo=1 )
    print( '          base bound [%.3f %.3f]' % bounds1 )

    bounds2 = mm1.kl_lse_bound( mm2, algo=2 )
    print( 'data-dependent bound [%.3f %.3f]' % bounds2 )

    improve = 1-(bounds2[1]-bounds2[0])/(bounds1[1]-bounds1[0])
    improve *= 100
    print( '%.0f%% improvement' % improve )

    _min = results.min( 1 )
    _max = results.max( 1 )
    mean = results.mean( 1 )
    std  = results.std( 1 )

    x = [ i+0.5 for i,_ in enumerate(sample_size_range) ]
    ax.axhline( bounds1[0], linewidth=2, alpha=.6, color='blue' )
    ax.axhline( bounds1[1], linewidth=2, alpha=.6, color='red'  )

    ax.axhline( bounds2[0], linewidth=2, alpha=.6, color='green',  ls='dashed' )
    ax.axhline( bounds2[1], linewidth=2, alpha=.6, color='orange', ls='dashed' )

    ax.errorbar( x, mean, yerr=std, fmt='o', color='black', elinewidth=2 )
    #xshifted = [ i+.2 for i in x ]
    #ax.errorbar( xshifted, mean, yerr=[mean-_min, _max-mean], fmt='none', color='blue' )

    ax.set_xticks( x )
    ax.set_xticklabels( [ r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$' ] )
    ax.set_xlim( 0, len(sample_size_range) )

    y_min = mean[0]-1.2*std[0]
    y_max = mean[0]+1.2*std[0]
    y_min = min( y_min, bounds1[0] - .1*(bounds1[1]-bounds1[0]) )
    y_max = max( y_max, bounds1[1] + .1*(bounds1[1]-bounds1[0]) )
    ax.set_ylim( y_min, y_max )

    blue   = mlines.Line2D( [], [], color='blue',   label='CELB' )
    red    = mlines.Line2D( [], [], color='red',    label='CEUB' )
    green  = mlines.Line2D( [], [], color='green',  label='CEALB', ls='dashed' )
    orange = mlines.Line2D( [], [], color='orange', label='CEAUB', ls='dashed' )
    ax.legend( handles=[blue, red,green,orange], fontsize=10, loc='best', fancybox=True, framealpha=0.5 )

    ax.set_title( title + ( ' (%.0f\%%)' % improve ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='estimate KL divergence'
                                      '/entropy of mixture models' )
    parser.add_argument( '--kl',      nargs=2, type=str )
    parser.add_argument( '--entropy', type=str )
    args = parser.parse_args()

    if args.kl:
        if os.access( args.kl[0], os.R_OK ) and os.access( args.kl[1], os.R_OK ):
            mm1 = load( args.kl[0] )
            mm2 = load( args.kl[1] )
            name1 = os.path.splitext( os.path.basename( args.kl[0] ) )[0]
            name2 = os.path.splitext( os.path.basename( args.kl[1] ) )[0]
            print( '%s:\n%s\n' % ( name1, mm1 ) )
            print( '%s:\n%s\n' % ( name2, mm2 ) )

            fig = plt.figure()

            show_signal( fig.add_subplot( 221 ), mm1, r'$\mathtt{%s}$' % name1.upper() )
            show_signal( fig.add_subplot( 222 ), mm2, r'$\mathtt{%s}$' % name2.upper() )

            show_kl( fig.add_subplot(223), mm1, mm2, r'KL($\mathtt{%s:%s}$)' % ( name1.upper(), name2.upper() ) )
            show_kl( fig.add_subplot(224), mm2, mm1, r'KL($\mathtt{%s:%s}$)' % ( name2.upper(), name1.upper() ) )

            ofilename = os.path.join( os.path.dirname( args.kl[0] ),
                                      'kl_%s_%s.pdf' % ( name1, name2 ) )
            fig.savefig( ofilename, bbox_inches='tight',
                         pad_inches=0, transparent=True )
            print( 'results saved to "%s"' %  ofilename )

        else:
            raise RuntimeError( 'cannot read the files for KL computation' )

    if args.entropy:
        if os.access( args.entropy, os.R_OK ):
            mm = load( args.entropy )
            name = os.path.splitext( os.path.basename( args.entropy ) )[0]
            print( '%s:\n%s\n' % ( name, mm ) )

            fig = plt.figure( figsize=(8,4) )
            show_signal(  fig.add_subplot( 121 ), mm, name.upper() )
            show_entropy( fig.add_subplot( 122 ), mm, r'Entropy($\mathtt{%s}$)' % name.upper() )

            ofilename = os.path.join( os.path.dirname( args.entropy ),
                                      'entropy_%s.pdf' % name )
            fig.savefig( ofilename, bbox_inches='tight',
                         pad_inches=0, transparent=True )

        else:
            print( 'cannot read the files for entropy computation' )

