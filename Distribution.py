from __future__ import print_function
from __future__ import division
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import erf, gamma, gammainc
import scipy.integrate as integrate

class Distribution( object ):
    '''
    general probability distribution
    '''
    __metaclass__ = ABCMeta

    def __init__( self ):
        '''
        common constants
        '''
        self.dtype = np.float64
        self.eps   = np.finfo( self.dtype ).eps

    ############### Interfaces ###############
    ##
    @abstractmethod
    def dim( self ):
        '''dimensionality'''
        raise NotImplementedError()

    @abstractmethod
    def key( self ):
        '''sort key'''
        raise NotImplementedError()

    @abstractmethod
    def loglikelihood( self, x ):
        '''compute loglikelihood of x
           as a Distribution object is used in a mixture model
           its weight is also considered in the computation

            x can be scalar on 1darray
        '''
        raise NotImplementedError()

    @abstractmethod
    def diff_loglikelihood( self, other, x ):
        '''
        self.loglikelihood(x) - other.loglikelihood(x)
        x is scalar
        this is for building evelopes
        '''
        raise NotImplementedError()

    @abstractmethod
    def draw( self, num_samples, chaos ):
        '''draw samples'''
        raise NotImplementedError()

    @abstractmethod
    def intersect( self, other ):
        '''compute intersection points'''
        raise NotImplementedError()

    @abstractmethod
    def prob( self, a, b ):
        '''probability mass from a to b
        left() <= a,b <= right()
        alpha is considered in the computation'''
        raise NotImplementedError()

    @abstractmethod
    def segment_cross( self, other, a, b ):
        '''
        \int self(x) \ln other(x) dx, x\in[a, b]
        alpha (weight) of both self and other
        is considered in the computation
        '''
        raise NotImplementedError()

    @abstractmethod
    def bound_ratio( self, other, a, b ):
        '''
        return the lower and upper bounds of the ratio
        self(x)/other(x), a<=x<=b

        either a or b can be infinity
        '''
        raise NotImplementedError()
    ##########################################

class Distribution1D( Distribution ):
    '''
    1D distributions
    '''

    __metaclass__ = ABCMeta

    def dim( self ):
        return 1

    @abstractmethod
    def left( self ):
        '''left bound of support'''
        raise NotImplementedError()

    @abstractmethod
    def right( self ):
        '''right bound of support'''
        raise NotImplementedError()

class Gaussian1D( Distribution1D ):
    '''
    univariate Gaussian
    '''
    def __init__( self, mu, sigma, alpha=1 ):
        super( self.__class__, self ).__init__()

        self.mu    = self.dtype( mu )

        self.sigma = self.dtype( sigma )
        if self.sigma <= 0: self.sigma = self.eps

        self.alpha = self.dtype( alpha )
        if self.alpha <= 0: self.alpha = self.eps

    def __str__( self ):
        return ( 'Gaussian(%5g, %5g) weight=%g' % ( self.mu,
                                                    self.sigma,
                                                    self.alpha ) )

    def key( self ):
        '''
        sort Gaussians
        from big sigma (high entropy)
        to small sigma (low entropy)
        '''
        return ( -self.sigma )

    def left( self ):
        return -np.inf

    def right( self ):
        return np.inf

    def const( self ):
        '''
        constant scalar for computing the loglikelihood
        '''
        return ( -.5*np.log( 2*np.pi )
                 -np.log( self.sigma )
                 +np.log( self.alpha ) )

    def loglikelihood( self, x ):
        '''
        compute the loglikelihood of x
        x can be a scalar, or 1darray of samples

        if x is inf, loglikelihood is -inf

        return a scalar or 1darray of the same shape
        '''
        return -.5 * ((x-self.mu)/self.sigma)**2 + self.const()

    def diff_loglikelihood( self, other, x ):
        if np.isinf( x ):
            if self.sigma < other.sigma:
                return -np.inf
            elif self.sigma > other.sigma:
                return np.inf
            elif self.mu < other.mu:
                return -x
            elif self.mu > other.mu:
                return x
            else:
                return self.const() - other.const()

        else:
            return self.loglikelihood(x)-other.loglikelihood(x)

    def draw( self, num_samples, chaos ):
        '''
        draw samples from the Gaussian
        chaos is RandomGenerator
        return a 1darray
        '''
        return chaos.normal( self.mu, self.sigma, num_samples )

    def intersect( self, other ):
        '''
        intersection point of log-likelihood function
        return a list of intersection points
        '''
        l1 = -.5 / (self.sigma**2)
        l2 = -.5 / (other.sigma**2)

        a = l1 - l2
        b = -2 * l1 * self.mu + 2 * l2 * other.mu
        c = l1 * (self.mu**2)  + self.const() \
          - l2 * (other.mu**2) - other.const()

        # solve ax^2 + bx + c = 0
        if a == 0 and b == 0:
            return []
        elif a == 0 and b != 0:
            return [ -c/b ]
        else:
            return self.__solve_quad( a, b, c )

    def __solve_quad( self, a, b, c ):
        '''
        ax^2 + bx + c = 0 (a!=0)
        '''
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return []
        elif discriminant == 0:
            return [ -.5 * b / a ]
        else:
            return [ -.5 * ( b - np.sqrt(discriminant) ) / a,
                     -.5 * ( b + np.sqrt(discriminant) ) / a ]

    def prob( self, a, b ):
        '''
        integrating the alpha-weighted Gaussian density (mu, sigma) from a to b
        '''
        return .5 * self.alpha * (
                      erf( ( b - self.mu ) / ( np.sqrt(2)*self.sigma ) )
                    - erf( ( a - self.mu ) / ( np.sqrt(2)*self.sigma ) ) )

    def segment_cross( self, other, a, b ):
        '''
        Integrate[ self(x) \ln other(x) ] from a to b
        '''
        _c = other.const() * self.prob( a, b )
        _c -= .5 * self.__int( other.mu, a, b ) / ( other.sigma**2 )
        return _c

    def __int( self, mu, a, b ):
        '''
        Integrate( self(x) (x-mu)^2 ) from a to b
        the sophisticated result is from WolframAlpha
        '''
        term1 = ( (mu-self.mu)**2 + self.sigma**2 ) * self.prob( a, b )

        if np.isinf( a ):
            term2 = 0
        else:
            term2 = self.sigma * self.alpha * (a+self.mu-2*mu) \
                    * np.exp( -.5*((a-self.mu)/self.sigma)**2 ) \
                    / np.sqrt(2*np.pi)

        if np.isinf( b ):
            term3 = 0
        else:
            term3 = -self.sigma * self.alpha * (b+self.mu-2*mu) \
                    * np.exp( -.5*((b-self.mu)/self.sigma)**2 ) \
                    / np.sqrt(2*np.pi)

        return term1+term2+term3

    def bound_ratio( self, other, a, b ):
        '''
        lower-upper bound of self(x)/other(x)
        where x is in the range [a,b]
        '''

        candi = [
                np.exp( self.diff_loglikelihood( other, a ) ),
                np.exp( self.diff_loglikelihood( other, b ) ),
                ]

        if self.sigma != other.sigma:
            x = ( self.mu / (self.sigma**2) - other.mu / (other.sigma**2) ) / \
                (     1.0 / (self.sigma**2) -      1.0 / (other.sigma**2) )
            if a < x and x < b:
                candi.append( np.exp( self.diff_loglikelihood(other, x) ) )

        return min( candi ), max( candi )

class Rayleigh1D( Distribution1D ):
    '''
    univariate Rayleigh Distribution
    '''
    def __init__( self, sigma, alpha=1 ):
        super( self.__class__, self ).__init__()

        self.sigma = self.dtype( sigma )
        if self.sigma <= 0: self.sigma = self.eps

        self.alpha = self.dtype( alpha )
        if self.alpha <= 0: self.alpha = self.eps

    def __str__( self ):
        return ( 'Rayleigh(%5g) weight=%g' % ( self.sigma,
                                               self.alpha ) )

    def key( self ):
        '''
        Rayleigh distributions will be sorted based on sigma
        '''
        return ( -self.sigma )

    def left( self ):
        return 0.0

    def right( self ):
        return np.inf

    def loglikelihood( self, x ):
        '''
        loglikelihood of x

        if x = 0,   return -inf
        if x = inf, return -inf
        '''
        x = np.array( x )
        ll = np.copy( x )
        ll[x==0]        = -np.inf
        ll[np.isinf(x)] = -np.inf

        idx     = np.logical_and( x>0, x<np.inf )
        ll[idx] = np.log(x[idx]) - (x[idx]**2) / (2*self.sigma**2) \
                  - 2 * np.log( self.sigma ) \
                  + np.log( self.alpha )
        return ll

    def diff_loglikelihood( self, other, x ):
        '''
        '''
        if x == 0:
            return - 2 * np.log( self.sigma ) + np.log( self.alpha ) \
                   + 2 * np.log( other.sigma ) - np.log( other.alpha )

        elif np.isinf( x ):
            if self.sigma < other.sigma:
                return -np.inf
            elif self.sigma > other.sigma:
                return np.inf
            else:
                return np.log( self.alpha ) - np.log( other.alpha )

        else:
            return self.loglikelihood(x)-other.loglikelihood(x)

    def draw( self, num_samples, chaos ):
        '''
        draw samples from the Rayleigh
        '''
        return chaos.rayleigh( self.sigma, num_samples )

    def intersect( self, other ):
        '''
        find non-trivial intersection in (0, inf)
        Note self(x) and other(x) must intersect at 0
        '''
        if self.sigma == other.sigma: return []

        # solve ax^2 = b
        a = .5/(self.sigma**2) - .5/(other.sigma**2)
        b = np.log(self.alpha)  - 2 * np.log(self.sigma) \
          - np.log(other.alpha) + 2 * np.log(other.sigma)
        squarex = b / a
        if squarex <= 0:
            return []
        else:
            return [ np.sqrt( squarex ), ]

    def prob( self, a, b ):
        '''
        probability mass from a to b
        '''
        l = -.5 / (self.sigma**2)
        return self.alpha * ( np.exp( l * a**2 ) - np.exp( l * b**2 ) )

    def segment_cross( self, other, a, b ):
        '''
        Integrate[ self(x) \ln other(x) ] from a to b
        '''

        _c = self.prob( a, b ) * ( np.log( other.alpha ) - 2 * np.log( other.sigma ) )

        # integrating the quadatic term
        # the expression is from WolframAlpha
        _c += -self.alpha * (2*self.sigma**2+a**2) \
                          * np.exp( -a**2/(2*self.sigma**2) ) \
                          / ( 2*other.sigma**2 )

        if not np.isinf( b ):
            _c += self.alpha * (2*self.sigma**2+b**2) \
                             * np.exp( -b**2/(2*self.sigma**2) ) \
                             / ( 2*other.sigma**2 )

        # numerically integrating the log(x) term
        def elnx( x ):
            return self.alpha * np.log( x ) * x * np.exp( -x**2/(2*self.sigma**2) ) / (self.sigma**2)
        _c += integrate.quad( elnx, a, b )[0]

        return _c

    def bound_ratio( self, other, a, b ):
        candi = [
                  np.exp( self.diff_loglikelihood( other, a ) ),
                  np.exp( self.diff_loglikelihood( other, b ) ),
                ]

        return min( candi ), max( candi )

class Exponential1D( Distribution1D ):
    '''
    Exponential distribution
    '''

    def __init__( self, rate, alpha=1 ):
        super( self.__class__, self ).__init__()

        self.rate = self.dtype( rate )
        if self.rate <= 0: self.rate = self.eps

        self.alpha = self.dtype( alpha )
        if self.alpha <= 0: self.alpha = self.eps

    def __str__( self ):
        return ( 'Exponential(%5g) weight=%g' % ( self.rate,
                                                  self.alpha ) )

    def key( self ):
        '''
        objects will be sorted based on their rates
        '''
        return self.rate

    def left( self ):
        return 0.0

    def right( self ):
        return np.inf

    def loglikelihood( self, x ):
        '''
        compute the loglikelihood of x

        if x = inf, return -inf
        '''
        return np.log( self.alpha * self.rate ) - self.rate * x

    def diff_loglikelihood( self, other, x ):
        if np.isinf( x ):
            if self.rate < other.rate:
                return np.inf
            elif self.rate > other.rate:
                return -np.inf
            else:
                return np.log( self.alpha * self.rate ) \
                     - np.log( other.alpha * other.rate )

        else:
            return self.loglikelihood(x)-other.loglikelihood(x)

    def draw( self, num_samples, chaos ):
        '''
        draw samples from the Exponential
        '''
        return chaos.exponential( 1.0/self.rate, num_samples )

    def intersect( self, other ):
        '''
        find intersection in [0, inf)
        '''
        if self.rate == other.rate: return []

        # solve ax = b s.t. x>=0
        a = self.rate - other.rate
        b = np.log( self.alpha * self.rate ) - np.log( other.alpha * other.rate )
        if b/a < 0:
            return []
        else:
            return [b/a,]

    def prob( self, a, b ):
        '''
        probability mass from a to b
        '''
        return self.alpha * ( np.exp( -self.rate * a )
                            - np.exp( -self.rate * b ) )

    def segment_cross( self, other, a, b ):
        _c = self.prob( a, b ) * np.log( other.alpha * other.rate )

        tmp = ( a * self.rate + 1 ) * np.exp( -a * self.rate )
        if not np.isinf( b ):
            tmp -= ( b * self.rate + 1 ) * np.exp( -b * self.rate )

        _c -= self.alpha * other.rate * tmp / self.rate 
        return _c

    def bound_ratio( self, other, a, b ):
        candi = [
                  np.exp( self.diff_loglikelihood( other, a ) ),
                  np.exp( self.diff_loglikelihood( other, b ) ),
                ]

        return min( candi ), max( candi )

class Gamma1D( Distribution1D ):
    '''
    Gamma distribution
    '''
    def __init__( self, k, theta, alpha=1 ):
        super( self.__class__, self ).__init__()

        self.k = self.dtype( k )
        if self.k <= 0: self.k = self.eps

        self.theta = self.dtype( theta )
        if self.theta <= 0: self.theta = self.eps

        self.alpha = self.dtype( alpha )
        if self.alpha <= 0: self.alpha = self.eps

    def __str__( self ):
        return ( 'Gamma(%5g, %5g) weight=%g' % ( self.k,
                                                 self.theta,
                                                 self.alpha ) )

    def key( self ):
        '''
        objects will be sorted based on their rates
        '''
        return -self.theta

    def left( self ):
        return 0.0

    def right( self ):
        return np.inf

    def const( self ):
        return np.log( self.alpha ) \
               - np.log( gamma(self.k) ) \
               - self.k * np.log( self.theta )

    def loglikelihood( self, x ):
        '''
        compute the loglikelihood of x

        if x = inf, return -inf
        '''

        if self.k == 1:
            ll = np.log( self.alpha ) - np.log( self.theta ) - x / self.theta

        else:
            x = np.array( x )
            ll = np.copy( x )
            ll[x==0] = np.sign(1-self.k) * np.inf
            try:
                ll[np.isinf(x)] = -np.inf
            except:
                print( x, ll, np.isinf(x) )

            idx     = np.logical_and( x>0, x<np.inf )
            ll[idx] = (self.k-1) * np.log( x[idx] ) \
                      - x[idx] / self.theta + self.const()
        return ll

    def diff_loglikelihood( self, other, x ):
        if x == 0:
            if self.k < other.k:
                return np.inf
            elif self.k > other.k:
                return -np.inf
            else:
                return self.const() - other.const()

        elif np.isinf( x ):
            if self.theta < other.theta:
                return -np.inf

            elif self.theta > other.theta:
                return np.inf

            elif self.k < other.k:
                return -np.inf

            elif self.k > other.k:
                return np.inf

            else:
                return self.const() - other.const()

        else:
            return self.loglikelihood(x) - other.loglikelihood(x)

    def draw( self, num_samples, chaos ):
        '''
        draw samples from the Gamma distribution
        '''
        return chaos.gamma( self.k, self.theta, num_samples )

    def intersect( self, other ):
        '''
        find intersection point of the log-likelihood function
        '''
        if self.k == other.k:
            # ax = b
            a = 1/self.theta - 1/other.theta
            b = self.const() - other.const()

            if a == 0:
                return []
            elif b/a < 0:
                return []
            else:
                return [ b/a, ]
        else:
            raise NotImplementedError()

    def prob( self, a, b ):
        if np.isinf( b ):
            # scipy's gammainc cannot correctly handle inf
            return self.alpha * ( 1 - gammainc( self.k, a/self.theta ) )

        else:
            return self.alpha * ( gammainc( self.k, b/self.theta )
                                - gammainc( self.k, a/self.theta ) )

    def segment_cross( self, other, a, b ):
        '''
        \sum_a^b self(x) \ln other(x) 
        '''

        # well, this is difficult to have closed form solution
        # so use numerical integration instead
        def _cross( x ):
            return np.exp( self.loglikelihood(x) ) * other.loglikelihood(x)

        return integrate.quad( _cross, a, b )[0]

    def bound_ratio( self, other, a, b ):
        candi = [
                  np.exp( self.diff_loglikelihood( other, a ) ),
                  np.exp( self.diff_loglikelihood( other, b ) ),
                ]

        if self.theta != other.theta:
            x = ( self.k - other.k ) / (1/self.theta - 1/other.theta)
            if a < x and x < b:
                candi.append( np.exp( self.diff_loglikelihood( other, x ) ) )

        return min( candi ), max( candi )

