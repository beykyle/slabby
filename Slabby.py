#!/usr/bin/env py36


"""
Slabby is a Diamond-Differenced, discrete ordinates, 1-D planar geometry, fixed-source, monoenergetic, isotropic scattering neutron transport code

"""

import numpy as np
import configparser
import sys
from scipy      import integrate
from matplotlib import pyplot as plt

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Slab:
  def __init__(self , numBins , width , *args , **kwargs):
    # initialize varaiables
    self.Q           = np.zeros(numBins)
    self.SigT        = np.zeros(numBins)
    self.SigS        = np.zeros(numBins)
    self.currentEps  = 0
    self.rho         = 0
    self.alpha       = 0 # diamond difference

    # get key word constructor arguments
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "loud" or key == "diagnostic" or key == "homogenous":
          setattr(self , key , booleanize(value))
        if key == "quadSetOrder" or key == "epsilon" or key =="out":
          setattr(self , key , value)
        elif key == "matData":
          self.getMatDataFromFile(value)

    # inital scalar flux guess
    self.scalarFlux = np.zeros(numBins)

  def setHomogenousData(self , sigt , sigs , q):
   for i in range(0,self.numBins-1):
     self.Q[i]    = q
     self.SigS[i] = sigs
     self.SigT[i] = sigt

  def getMatDataFromFile(self , filename):
   # alternative to homogenous data
   raise NotImplementedError

  def writeOutput(self , filename):
    if self.currentEps > self.epsilon and self.loud == False and self.diagnostic == True:
      # write the diagnostics to the output file
      with open(filename , "a") as output:
        output.write("Epsilon: " + '{:1.4f}'.format(self.currentEps) + " , Rho: " + '{:1.4f}'.format(self.rho) + "\r\n")
    elif self.currentEps > self.epsilon and self.loud == True and self.diagnostic == True:
      # print the diagnostics to the command line
      print("Epsilon: " + '{:1.4f}'.format(self.currentEps) + " , Rho: " + '{:1.4f}'.format(self.rho) + "\r\n")
    else:
      # the simulation is done, write the scalar flux to the output file
      with open(filename , "a") as output:
        output.write("\r\n Scalar Flux: \r\n")
        for i , val in enumerate(self.scalarFlux):
          output.write('{:1.4f}'.format( i * self.width / self.numBins ) + "   " + '{:1.7f}'.format(val) )

  def plotScalarFlux(self, iterNum ):
    x = np.linspace(0 , width , self.numbins)
    plt.plot(x , self.scalarFlux , "k*")
    plt.xlabel(r"$x$ [cm]")
    plt.ylabel(r"scalar flux, $\Phi$ [$\text{cm}^{-2} \text{s}^{-1} $]")

    if self.loud == True:
      plt.show()
    else:
      plt.saveplot(" flux_" + str(iterNum) + ".png")

  def setRightBoundaryFlux(self , rightFlux , *args , **kwargs):
    self.rightFlux = rightFlux
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "boundaryType":
          setattr(self , "rightBoundaryType" , value)

  def setLeftBoundaryFlux(self , leftFlux , *args , **kwargs):
    self.leftFlux = leftFlux
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "boundaryType":
          setattr(self , "leftBoundaryType" , value)

  def getScalarFlux(psi):
      return( integrate(psi) )

  def run(self):
    psiIn  = np.zeros(quadSetOrder)
    psiOut = np.zeros(quadSetOrder)
    while(self.currentEps > self.epsilon):
      pass
      # set inital angular flux on the left from the left boundary condition
      # initialize Sn isotropic source using the old scalar flux
      phi = getScalarFlux(psiIn)
      # run a transport sweep left to right
      # set inital angular flux on the right from the right boundary condition
      # run a transport sweep right to left
      # calculate new rho estimate
      # calculate new epsilon to test convergence


def booleanize(value):
    """Return value as a boolean."""
    true_values  = ("yes", "true", "1")
    false_values = ("no", "false", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_values:
        return True
    elif value.lower() in false_values:
        return False
    raise TypeError("Cannot booleanize ambiguous value '%s'" % value)

if __name__ == '__main__':

  # Read settings from input file
  inputFile = sys.argv[1]
  conf = configparser.ConfigParser()
  conf.read(inputFile)

  if 'General' in conf:
    loud          =  booleanize( conf['General']['Loud'].strip()        )
    diagnostic    =  booleanize( conf['General']['Diagnostic'].strip()  )
    outputFi      =              conf['General']['Output'].strip()
    epsilon       =  float(      conf['General']['convergence'].strip()  )
    quadSetOrder  =  int(        conf['General']['quadSetOrder'].strip() )
    if quadSetOrder % 2 != 0:
      print("Order of the quad set must be even! Exiting!")
      sys.exit()

  else:
    raise
    print("No General section found in input file! Exiting! ")
    sys.exit()

  if 'Slab' in conf:
    if 'matData' in conf['Slab']:
      print("non-homogenous data not yet implemented!")
      raise NotImplementedError
      sys.exit()
    else:
      width         =              conf['Slab']['width'].strip()
      homogenous    =  booleanize( conf['Slab']['homogenous'].strip() )
      bins          =  int(        conf['Slab']['int'].strip()        )
      SigT          =  float(      conf['Slab']['SigT'].strip()       )
      SigS          =  float(      conf['Slab']['SigS'].strip()       )
      Q             =  float(      conf['Slab']['Q'].strip()          )
  else:
    raise
    print("No Slab section found in input file! Exiting! ")
    sys.exit()

  if 'Boundary Conditions' in conf:
    if 'leftFlux' in conf['Boundary Conditions']:
      leftFlux  = float( conf['Boundary Conditions']['leftFlux'].strip() )
      rightFlux = float( conf['Boundary Conditions']['rightFlux'].strip() )
    else:
      leftFlux , rightFlux = 0 , 0

    left   =   conf['Boundary Conditions']['left'].strip()
    right  =   conf['Boundary Conditions']['right'].strip()
  else:
    raise
    print("No Slab section found in input file! Exiting! ")
    sys.exit()

  # Create slab object and run the simulation
  slab = Slab(bins , width , loud=loud , diagnostic=diagnostic , quadSetOrder=quadSetOrder , epsilon=epsilon , out=outputFi)
  if homogenous == True:
    slab.setHomogenousData(SigT , SigS , Q)
  else:
    raise NotImplementedError

  slab.setRightBoundary(rightFlux , boundaryType = right)
  slab.setLeftBoundary(  leftFlux , boundaryType = left )
  slab.run()


