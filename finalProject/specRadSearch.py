#!/usr/bin/env py36

"""
specRadSearch.py is a script to numerically search the spectral radius for the space of

"""
from fractions import Fraction
import numpy as np
from scipy.special import roots_legendre
from matplotlib    import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

mu , weights = roots_legendre(16 , mu=False)
weights = np.array(weights)
weights = weights / sum(weights)

def omega_n( tau , s ,  m ):
  return( (np.cos(tau)**2 - 3 * m**2 ) /
          (np.cos(tau)**2 +  ( 2 * np.sin(tau) / s )**2 * m**2 ) )

def omega( tau , s ):
  val  = 0
  for m , wgt in zip(mu , weights):
    val = val + omega_n( tau , s , m ) * wgt

  return(val)

if __name__ == '__main__':
    tau   = np.linspace(0, 2 * np.pi, 500)
    SigTh = [ 3/n for n in range(1,16) ]

    for i ,  s in enumerate(SigTh):
      o = omega(tau , s)
      plt.plot( tau , o , label=r"$\Sigma_t h =$ 3/" + str(i+1) )
      plt.xlabel(r"$\tau$")
      plt.ylabel("$\omega$")
      plt.xlim([0 , 2 * np.pi])
      xx = [i for i , j in enumerate(o) if j == max(o) ]
      plt.plot( tau[xx[0]] , max(o) , 'dr' , label=r"$|\omega{(\tau , 3/" + str(i+1)  + " )}|_\infty$" )
      #plt.title(r"$\Sigma_t h =$ 3/" + str(i+1) )
      plt.tight_layout()
      plt.legend()
      plt.savefig( str(i) + ".png"  )
      plt.cla()
#      plt.show()




