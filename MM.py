from __future__ import print_function
from __future__ import division
from abc import ABCMeta, abstractmethod

from Distribution import Gaussian1D, Rayleigh1D, Exponential1D, Gamma1D
from scipy.misc import logsumexp
import numpy as np
import sys, os

class MM( object ):
    '''
    mixture model
    '''
    __metaclass__ = ABCMeta

    ##############################################################
    # Interfaces
    @abstractmethod
    def _update_upper_envelope( self, intervals, idx ):
        '''
        update the upper envelope given by intervals
        based on self.dist[idx]
        this method is used to build the upper-envelope of an MM

        return the updated envelope
        '''
        raise NotImplementedError()

    @abstractmethod
    def _update_lower_envelope( self, intervals, idx ):
        '''
        update the lower envelope of mixtures
        '''
        raise NotImplementedError()

    @abstractmethod
    def plot_left( self ):
        '''
        xmin for plots
        '''
        raise NotImplementedError()

    @abstractmethod
    def plot_right( self ):
        '''
        xmax for plots
        '''
        raise NotImplementedError()

    @staticmethod
    def load( filename ):
        '''
        static method
        build a MM object from text file
        '''
        raise NotImplementedError()

    ##############################################################

    @staticmethod
    def _readtxt( filename ):
        '''
        static method
        iterate all number lines in a txt file
        '''
        if not os.access( filename, os.R_OK ):
            raise RuntimeError( 'cannot read file %s' % filename )

        for line in open( filename ):
            line = line.strip()
            if not line: continue

            # ignore all comment lines starting with #
            if line.startswith('#'): continue

            yield [ float(f) for f in line.split() ]

    def __init__( self, components, random_seed=None, filename='' ):
        '''
        components is a list of Distributions
        random_seed is to initialize the random generator
        '''
        self.dtype = np.float64
        self.eps   = np.finfo( self.dtype ).eps

        self.dirname  = os.path.dirname( filename )
        self.basename = os.path.basename( filename )

        # an MM object has its own random generator
        self.chaos = np.random.RandomState( random_seed )

        # sort the components based on distibutions's key() method
        assert( len(components) > 0 )
        self.dist = sorted( components, key=lambda c:c.key() )

        # ensure all components have same dimension
        _ndim = self.dist[0].dim()
        for _d in self.dist[1:]: assert( _ndim == _d.dim() )

        # Important: re-normalize the weights
        alpha = np.array( [ _d.alpha for _d in self.dist ] )
        alpha /= alpha.sum() + self.eps
        for _a, _d in zip( alpha, self.dist ): _d.alpha = _a

    def __str__( self ):
        '''
        string representation
        '''
        return '\n'.join( [ str(_d) for _d in self.dist ] )

    def dim( self ):
        return self.dist[0].dim()

    def left( self ):
        return self.dist[0].left()

    def right( self ):
        return self.dist[0].right()

    def seed( self, random_seed ):
        '''
        reseed the internal random generator
        '''
        self.chaos.seed( random_seed )

    def draw( self, num_samples ):
        '''
        draw samples from the mixture model
        '''
        alpha   = np.array( [ _d.alpha for _d in self.dist ] )
        counts  = self.chaos.multinomial( num_samples, alpha )
        samples = [ _d.draw( _c, self.chaos )
                    for _c,_d in zip( counts, self.dist ) ]

        if samples[0].ndim == 1:
            return np.hstack( samples )
        if samples[0].ndim == 2:
            return np.vstack( samples )
        else:
            raise NotImplementedError()

    def comp_ll( self, x, idx ):
        '''
        component-wise
        log-likelihood of x under dist[idx]
        '''
        if ( np.size( x ) == 1 ) and np.isinf( x ):
            return 0
        else:
            return self.dist[idx].loglikelihood( x )

    def loglikelihood( self, x ):
        '''
        compute the loglikelihood of x
        x can be a scalar (one sample)
        or 1darray of samples
        or 2darray with one sample per row
        '''
        ll = np.array( [ self.comp_ll( x, i )
                         for i, _d in enumerate( self.dist ) ] )
        return logsumexp( ll, axis=0 )

    def kl_sampling( self, other, num_samples ):
        '''
        measure the KL divergence KL(self; other)
        based on random sampling
        '''
        samples = self.draw( num_samples )
        return np.mean( self.loglikelihood( samples )
                     - other.loglikelihood( samples ) )

    def entropy_sampling( self, num_samples ):
        '''
        measure the entropy by random sampling
        '''
        samples = self.draw( num_samples )
        return -np.mean( self.loglikelihood( samples ) )

    def kl_lse_bound( self, other, algo=1 ):
        '''
        KL divergence based on LSE bounding

        algo = 1 basic bound
        algo = 2 data-dependent bound
        '''
        lb1, ub1 = self.__cross( self,  algo )
        lb2, ub2 = self.__cross( other, algo )
        return lb1 - ub2, ub1 - lb2

    def entropy_lse_bound( self, algo=1 ):
        '''
        Entropy based on LSE bounding
        '''
        lb, ub = self.__cross( self, algo )
        return -ub, -lb

    def __cross( self, other, algo ):
        '''
        compute \int self(x) \ln other(x) dx
        the negative cross entropy
        '''
        lb = 0
        ub = 0

        other_env = other.envelope()
        for _d in self.dist:
            for begin, end, u_idx, l_idx in other_env:
                tmp_u = _d.segment_cross( other.dist[u_idx], begin, end )
                tmp_l = _d.segment_cross( other.dist[l_idx], begin, end )
                prob  = _d.prob( begin, end )

                if algo == 1:
                    # basic bounds
                    lb += max( tmp_u, tmp_l + prob * np.log( len(other.dist) ) )
                    ub += tmp_u + prob * np.log( len(other.dist) )

                elif algo == 2:
                    # data-dependent bounds
                    uratio = np.array(
                              [ other_d.bound_ratio( other.dist[u_idx], begin, end )
                                for i, other_d in enumerate( other.dist )
                                if i != u_idx ] )
                    uratio = np.log( 1 + uratio.sum(0) )

                    lb += max( tmp_u + prob * uratio[0],
                               tmp_l + prob * np.log( len(other.dist) ) )
                    ub += tmp_u + prob * uratio[1]

        return lb, ub

    def upper_envelope( self ):
        '''
        only for 1d mixture models

        partition the real line into several intervals,
        so that one component will dominate one interval

        return a list of ( begin, end, idx )
        where ( begin, end ) is an interval
        idx is the index of the dominator component
        '''

        assert( self.dim() == 1 )

        # initially, component 0 dominate the whole support
        intervals = [ ( self.left(), self.right(), 0 ) ]

        # update the dominators
        for idx in range( 1, len(self.dist) ):
            intervals = self._update_upper_envelope( intervals, idx )

        return intervals

    def lower_envelope( self ):
        '''
        only for 1d mixture models

        partition the real line into several intervals,
        so that one component is the smallest density in one interval

        return a list of ( begin, end, idx )
        where ( begin, end ) is an interval
        idx is the index of the dominator component
 
        '''

        assert( self.dim() == 1 )

        # initially, component -1 dominate the whole support
        intervals = [ ( self.left(), self.right(), len(self.dist)-1 ) ]

        # update the dominators
        for idx in range( len(self.dist)-1 )[::-1]:
            intervals = self._update_lower_envelope( intervals, idx )

        return intervals

    def envelope_filename( self ):
        if self.basename:
            root, ext = os.path.splitext( self.basename )
            return os.path.join( self.dirname, root + '.envelope' )
        else:
            return ''

    def envelope( self ):
        '''
        build envelope
        split the support into several fundemental intervals,
        so that each interval has unique upper/lower components

        return a list of ( begin, end, upper, lower )
        '''
        assert( self.dim() == 1 )
        env = []

        if os.access( self.envelope_filename(), os.R_OK ):
            if os.path.getmtime( self.envelope_filename() ) >= \
               os.path.getmtime( os.path.join( self.dirname, self.basename ) ):
            # if there is an envelope file newer than the mixture file
            # then directly load the envelope

                for a, b, c, d in self._readtxt( self.envelope_filename() ):
                    env.append( ( a, b, int(c), int(d) ) )

        else:
            u = self.upper_envelope()
            l = self.lower_envelope()

            start = self.left()
            while start < self.right():
                if len( u ) > 0:
                    _tmp, end_u, u_idx = u[0]
                if len( l ) > 0:
                    _tmp, end_l, l_idx = l[0]

                if end_u < end_l:
                    u.pop(0)
                else:
                    l.pop(0)

                env.append( ( start, min(end_u, end_l), u_idx, l_idx ) )
                start = min( end_u, end_l )

            if self.envelope_filename():
                with open( self.envelope_filename(), 'w' ) as of:
                    of.write( "# start end upper lower\n" )
                    for a, b, c, d in env:
                        of.write( '%f %f %d %d\n' % (a,b,c,d) )

        return env

    #############################################################
    # Utility functions
    #############################################################

    def _search_location( self, value, intervals ):
        '''
        binary searching of intervals that contains value
        '''
        begin = 0               # inclusive
        end   = len(intervals)  # not invlusive

        while begin < end:
            candi = ( begin + end ) // 2
            side = self.__side( value, intervals[candi] )
            if side == -1:
                end   = candi
            elif side == 1:
                begin = candi + 1
            else:
                break

        assert( self.__side( value, intervals[candi] ) == 0 )
        return candi

    def __side( self, value, interval ):
        '''
        determine value on which side of the interval
        return
        -1 if value is on left
         1 if value is on right
         0 if value is inside the interval
        '''

        if value < interval[0]:
            return -1
        elif value > interval[1]:
            return 1
        else:
            return 0

    def _cut( self, intervals, idx ):
        '''
        binary searching the left break point

        we already know that commponent[idx] intersects
        at most once with intervals, return L, R
        where L is intervals on the left of the break point,
        and R is the intervals on the right

        if there is no intersection, return L, L
        where L is the updated intervals
        '''

        break_l = self.__break( idx, intervals[0]  )
        break_r = self.__break( idx, intervals[-1] )

        if break_l * break_r == 1:
            self.cut_pt = None

            cover = [ (intervals[0][0], intervals[-1][1], idx ) ]
            return cover, cover

        elif break_l == 0:
            candi = 0

        elif break_r == 0:
            candi = len(intervals)-1

        else:
            begin = 1
            end   = len( intervals )-1

            while begin < end:
                candi = ( begin + end ) // 2
                breaked = self.__break( idx, intervals[candi] )
                if breaked == break_l:
                    begin = candi + 1
                elif breaked == break_r:
                    end = candi
                else:
                    break

        # now we know that idx intersect the evelope at intervals[candi]
        begin, end, domi = intervals[candi]
        l = [ p
              for p in self.dist[idx].intersect( self.dist[domi] )
              if self.__side( p, intervals[candi] ) == 0 ]

        assert( len(l) == 1 )
        self.cut_pt = l[0]

        return ( intervals[:candi] + [ ( begin, l[0], domi ) ],
                 [ (l[0], end, domi) ] + intervals[(candi+1):] )

    def __break( self, idx, interval ):
        '''
        check if component[idx] breaks the interval
        in the sense that
        f(x):= self.dist[idx]( x )
        and
        g(x):= self.dist[domi]( x )
        intersect inside the interval

        return 0 if break
        return 1  if idx is above current dominator
        return -1 if idx is below current dominator
        '''
        begin, end, domi = interval

        if np.isinf( begin ) or np.isinf( end ):
            l = [ p for p in self.dist[idx].intersect( self.dist[domi] )
                  if self.__side( p, interval ) == 0 ]

            if len( l ) == 0:
                if not np.isinf( begin ):
                    x = begin + 1
                elif not np.isinf( end ):
                    x = end - 1
                else:
                    x = 0

                return np.sign( self.dist[idx].diff_loglikelihood( self.dist[domi], x ) )
            else:
                return 0

        bdiff = self.dist[idx].diff_loglikelihood( self.dist[domi], begin )
        ediff = self.dist[idx].diff_loglikelihood( self.dist[domi], end )

        if ( bdiff * ediff <= 0 ):
            return 0
        elif bdiff > 0:
            return 1
        else:
            return -1

    def _simplify( self, intervals ):
        '''
        merge intervals. if two nearby intervals
        share the same dominator, then merge them
        '''
        simplified = []
        itr = 0
        while itr < len( intervals ):
            domi = intervals[itr][-1]
            span = itr+1
            while( span < len( intervals ) and intervals[span][-1] == domi ):
                span += 1

            simplified.append( ( intervals[itr][0], intervals[span-1][1], domi ) )
            itr = span

        return simplified

class GMM( MM ):
    '''
    Gaussian Mixture Model
    '''
    def _update_upper_envelope( self, intervals, idx ):
        '''
        update the partitions based on i'th component
        '''

        # find component[idx].mu is in which interval 
        mu = self.dist[idx].mu
        loc = self._search_location( mu, intervals )
        begin, end, domi = intervals[loc]

        # idx didn't break out the envelope
        if self.comp_ll( mu, idx ) <= self.comp_ll( mu, domi ):
            return intervals

        # split intervals into two half
        left_intervals  = intervals[:loc] + [ ( begin, mu, domi ) ]
        right_intervals = [ (mu, end, domi) ] + intervals[(loc+1):]

        l, tmp = self._cut(  left_intervals,  idx )
        tmp, r = self._cut( right_intervals, idx )

        return self._simplify( l + [ ( l[-1][1], r[0][0], idx ) ] + r )

    def _update_lower_envelope( self, intervals, idx ):
        '''
        update the lower envelope of Gaussian densities
        equivalently, upper envelope of parabolas
        '''

        # find component[idx].mu is in which interval 
        if len( intervals ) == 1:
            self.low_pt = self.dist[intervals[0][-1]].mu

        loc = self._search_location( self.low_pt, intervals )
        begin, end, domi = intervals[loc]

        # idx didn't break out the envelope
        if self.comp_ll( self.low_pt, idx ) >= self.comp_ll( self.low_pt, domi ):
            return intervals

        # split intervals into two half
        left_intervals  = intervals[:loc] + [ ( begin, self.low_pt, domi ) ]
        right_intervals = [ (self.low_pt, end, domi) ] + intervals[(loc+1):]

        l, tmp = self._cut(  left_intervals, idx )
        l_cut_pt = self.cut_pt

        tmp, r = self._cut( right_intervals, idx )
        r_cut_pt = self.cut_pt

        if ( l_cut_pt is None or self.dist[idx].mu > l_cut_pt ) \
           and \
           ( r_cut_pt is None or self.dist[idx].mu < r_cut_pt ):
            self.low_pt = self.dist[idx].mu

        else:
            if l_cut_pt and self.comp_ll( l_cut_pt, idx ) >= self.comp_ll( self.low_pt, idx ):
                self.low_pt = l_cut_pt

            if r_cut_pt and self.comp_ll( r_cut_pt, idx ) >= self.comp_ll( self.low_pt, idx ):
                self.low_pt = self.cut_pt

        return self._simplify( l + [ ( l[-1][1], r[0][0], idx ) ] + r )

    def plot_left( self ):
        return min( [_d.mu-3*_d.sigma for _d in self.dist] )

    def plot_right( self ):
        return max( [_d.mu+3*_d.sigma for _d in self.dist] )

    @staticmethod
    def load( filename ):
        components = [ Gaussian1D( mu, sigma, alpha=alpha )
                       for alpha, mu, sigma in MM._readtxt( filename ) ]
        return GMM( components, filename=filename )

    def entropy_upper_bound( self ):
        moment1 = sum( [ _d.alpha * _d.mu for _d in self.dist ] )
        moment2 = sum( [ _d.alpha * (_d.mu**2 + _d.sigma**2) for _d in self.dist ] )
        var = moment2 - moment1**2

        return .5 * np.log( 2*np.pi*np.e*var )

class RMM( MM ):
    '''
    Rayleigh Mixture
    '''
    def _update_upper_envelope( self, intervals, idx ):
        '''
        update the upper envelope
        '''
        begin, end, domi = intervals[0]

        # idx didn't break out the envelope
        if self.dist[idx].diff_loglikelihood( self.dist[domi], 0 ) <= 0:
            return intervals

        # idx will become intervlas[0]
        tmp, r = self._cut( intervals, idx )
        return self._simplify( [ ( 0, r[0][0], idx ) ] + r )

    def _update_lower_envelope( self, intervals, idx ):
        '''
        update the lower envelope
        '''
        begin, end, domi = intervals[0]

        # idx didn't break out the envelope
        if self.dist[idx].diff_loglikelihood( self.dist[domi], 0 ) >= 0:
            return intervals

        # idx will become intervlas[0]
        tmp, r = self._cut( intervals, idx )
        return self._simplify( [ ( 0, r[0][0], idx ) ] + r )

    def plot_left( self ):
        return 0

    def plot_right( self ):
        return max(  [ 3*_d.sigma for _d in self.dist ] )

    @staticmethod
    def load( filename ):
        components = [ Rayleigh1D( sigma, alpha=alpha )
                       for alpha, sigma in MM._readtxt( filename ) ]
        return RMM( components, filename=filename )

class EMM( MM ):
    '''
    Mixture of Exponential distributions
    '''
    def _update_upper_envelope( self, intervals, idx ):
        begin, end, domi = intervals[0]

        if self.dist[idx].diff_loglikelihood( self.dist[domi], 0 ) <= 0:
            return intervals

        tmp, r = self._cut( intervals, idx )
        return self._simplify( [ ( 0.0, r[0][0], idx ) ] + r )

    def _update_lower_envelope( self, intervals, idx ):
        begin, end, domi = intervals[0]

        if self.dist[idx].diff_loglikelihood( self.dist[domi], 0 ) >= 0:
            return intervals

        tmp, r = self._cut( intervals, idx )
        return self._simplify( [ ( 0.0, r[0][0], idx ) ] + r )

    def plot_left( self ):
        return 0

    def plot_right( self ):
        return max(  [ 1/_d.rate for _d in self.dist ] )

    @staticmethod
    def load( filename ):
        components = [ Exponential1D( rate, alpha=alpha )
                       for alpha, rate in MM._readtxt( filename ) ]
        return EMM( components, filename=filename )

class GaMM( MM ):
    '''
    Gamma Mixture Model
    All components must have the same k
    '''
    def _update_upper_envelope( self, intervals, idx ):
        '''
        same algorithm as EMM
        '''
        begin, end, domi = intervals[0]

        if self.dist[idx].diff_loglikelihood( self.dist[domi], 0.0 ) <= 0:
            return intervals

        tmp, r = self._cut( intervals, idx )
        return self._simplify( [ (0.0, r[0][0], idx ) ] + r )

    def _update_lower_envelope( self, intervals, idx ):
        '''
        same algorithm as EMM
        '''
        begin, end, domi = intervals[0]

        if self.dist[idx].diff_loglikelihood( self.dist[domi], 0.0 ) >= 0:
            return intervals

        tmp, r = self._cut( intervals, idx )
        return self._simplify( [ (0.0, r[0][0], idx ) ] + r )

    def plot_left( self ):
        return 0

    def plot_right( self ):
        return max(  [ 3*np.sqrt(_d.k)*_d.theta for _d in self.dist ] )

    @staticmethod
    def load( filename ):
        components = [ Gamma1D( k, theta, alpha=alpha )
                       for alpha, k, theta in MM._readtxt( filename ) ]
        return GaMM( components, filename=filename )

