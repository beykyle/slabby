#!/usr/bin/env py36


"""
Slabby is a Diamond-Differenced, discrete ordinates, 1-D planar geometry, fixed-source, monoenergetic, isotropic scattering neutron transport code

"""

import numpy as np
import configparser
import sys
from scipy.special import roots_legendre
from matplotlib    import pyplot as plt

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Slab:
  def __init__(self , numBins , width , *args , **kwargs):
    print("Intializing slab!")
    # initialize varaiables
    self.numBins     = int(numBins)
    self.binWidth    = float(width / numBins)
    self.width      =  float(width)
    self.Q           = np.zeros(numBins)
    self.SigT        = np.zeros(numBins)
    self.SigS        = np.zeros(numBins)
    self.currentEps  = 1000 # a big number
    self.rho         = 0

    print("number of bins: " + str(numBins))
    print("slab width : "    + str(width))

    # get key word constructor arguments
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "loud" or key == "diagnostic" or key == "homogenous":
          print(key + ": " + str(value))
          setattr(self , key , booleanize(value))
        elif key == "epsilon" or key == "alpha":
          print(key + ": " + str(value))
          setattr(self , key , float(value))
        elif key == "quadSetOrder":
          print(key + ": " + str(value))
          setattr(self , key , int(value))
        elif key == "matData":
          print(key + ": " + str(value))
          self.getMatDataFromFile(value)
        else:
          print(key + ": " + str(value))
          setattr(self , key , value)


    # inital scalar flux guess
    self.scalarFlux = np.ones(numBins) * 10.0

    # initialize quad set weights ,  make sure they're normalized
    self.mu , self.weights = roots_legendre(self.quadSetOrder , mu=False)
    self.weights = np.array(self.weights)
    self.weights = self.weights / sum(self.weights)

  def setHomogenousData(self , sigt , sigs , q):
   for i in range(0,self.numBins-1):
     self.Q[i]    = q
     self.SigS[i] = sigs
     self.SigT[i] = sigt

  def getMatDataFromFile(self , filename):
   # alternative to homogenous data
   raise NotImplementedError

  def writeOutput(self , filename):
    # write the diagnostics to the output file
    with open(filename , "a") as output:
      output.write("Epsilon: " + '{:1.8f}'.format(self.currentEps) + " , Rho: " + '{:1.8f}'.format(self.rho) + "\r\n")
    # print the diagnostics to the command line
    print("Epsilon: " + '{:02.4f}'.format(self.currentEps) + " , Rho: " + '{:02.4f}'.format(self.rho) + "\r\n")

  def plotScalarFlux(self, iterNum ):
    plt.ion()
    x = np.linspace(0 , self.width , self.numBins)
    plt.plot(x , self.scalarFlux , "k*")
    plt.xlabel(r"$x$ [cm]")
    plt.ylabel(r"scalar flux, $\Phi$ [cm$^{-2}$ s$^{-1}$ ]")
    plt.title("Iteration " + str(iterNum))

    if self.loud == True:
      plt.draw()
      plt.pause(0.001)
    if self.currentEps < self.epsilon:
      plt.show()
    if diagnostic == True:
      plt.savefig("./flux_" + str(iterNum) + ".png")

  def setRightBoundaryFlux(self , rightFlux , *args , **kwargs):
    self.rightFlux = rightFlux
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "boundaryType":
          setattr(self , "rightBoundaryType" , value)
    self.rightBoundaryFlux = np.ones(int(self.quadSetOrder / 2)) * rightFlux
    if self.rightBoundaryType == "Isotropic":
      for i , val in self.rightBoundaryFlux:
        self.rightBoundaryFlux[i] = rightFlux / (4 * np.pi) * self.weights[i]

  def setLeftBoundaryFlux(self , leftFlux , *args , **kwargs):
    self.leftFlux = leftFlux
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "boundaryType":
          setattr(self , "leftBoundaryType" , value)
    self.leftBoundaryFlux = np.ones( int(self.quadSetOrder / 2)) * leftFlux
    if self.leftBoundaryType == "Isotropic":
      for i , val in self.leftBoundaryFlux:
        self.leftBoundaryFlux[i] = leftFlux / (4 * np.pi) * self.weights[i]

  def getScalarFlux(self , psi , *args , **kwargs):
    # integrate the angular flux in a spatial bin according to the Gauss-Legendre quad sets
    offset = 0
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "direction" and value == "right":
          offset = self.quadSetOrder / 2

    phi = 0
    # integrate psi over the quadrature set
    for i , val in enumerate(psi):
      phi += val * self.weights[i + offset]
    return(phi)

  def getScatterSource(self):
    # for an isotropically scattering problem we dont need to calculate any
    # angular moments of the scalar flux
    return( 0.5 * ( np.multiply(self.SigS , self.scalarFlux) + self.Q ) )

  def transportSweep(self):
    N = int(self.quadSetOrder / 2)
    # this transport function explicitly iterates through the spatial variable
    # and is vectorized in the angular variable for SIMD optimization

    # precompute scatter source for the isotropic problem (only discretized over space)
    Sn = self.getScatterSource()

    # find the left boundary flux
    psiIn = self.leftBoundaryFlux

    # initialize variables used in the sweep
    psiAv  = np.zeros(( int( self.quadSetOrder / 2)))
    psiOut = np.zeros(( int( self.quadSetOrder / 2)))

    # sweep left to right
    for i in range(0,self.numBins - 1):
      # find the upstream flux at the right bin boundary
      psiOut = np.divide( (np.multiply( self.c1lr[i,:] , psiIn[:]  ) + Sn[i] *  self.binWidth ) , self.c2lr[i,:])
      # find the average flux in the bin according to the spatial differencing scheme
      psiAv  = (1 + self.alpha) * 0.5 * psiOut[:] + (1 - self.alpha) * 0.5 * psiIn[:]
      # find the scalar flux in this spatial bin from right-moving flux
      self.scalarFlux[i] = self.getScalarFlux(psiAv , direction=right)
      # set the incident flux on the next bin to exiting flux from this bin
      psiIn = psiOut

    # find the right boundary flux
    psiIn = self.rightBoundaryFlux

    # sweep right to left
    for i in range(self.numBins -1 , 0 , -1):
      # find the upstream flux at the left bin boundary
      psiOut = np.divide( (np.multiply( self.c1rl[i,:] , psiIn[:]  ) + Sn[i] * self.binWidth ) , self.c2rl[i,:] )
      # find the average flux in the bin according to the spatial differencing scheme
      psiAv  = (1 + self.alpha) * 0.5 * psiOut[:] + (1 - self.alpha) * 0.5 * psiIn[:]
      # find the scalar flux in this spatial bin from left-moving flux
      self.scalarFlux[i] += self.getScalarFlux(psiAv , direction=left)
      # set the incident flux on the next bin to exiting flux from this bin
      psiIn = psiOut

  def estimateRho(self , oldScalarFlux):
    self.rho = np.dot(self.scalarFlux , self.scalarFlux) / np.dot(oldScalarFlux , oldScalarFlux)

  def testConvergence(self , oldScalarFlux):
    self.currentEps = max( np.divide( np.abs(self.scalarFlux - oldScalarFlux)  ,  np.abs(self.scalarFlux) + 0.000001 ) )

  def clearOutput(self):
    with open(self.out , "w") as outt:
      outt.write("Running 1-D transport!")

  def run(self):
    iterationNum = 0
    self.clearOutput()

    # precompute coefficients for solving for upstream
    # coefficients form a constant matrix, discretized over both angle and space
    N = int(self.quadSetOrder / 2)
    self.c1lr = np.zeros((self.numBins , int(self.quadSetOrder / 2)))
    self.c2lr = np.zeros((self.numBins , int(self.quadSetOrder / 2)))
    self.c1rl = np.zeros((self.numBins , int(self.quadSetOrder / 2)))
    self.c2rl = np.zeros((self.numBins , int(self.quadSetOrder / 2)))

    for i in range(0 , self.numBins ):
      # left to right
      self.c1lr[i,:] = (self.mu[N:] - (1 - self.alpha) * self.SigT[i] * self.binWidth / 2)[:]
      self.c2lr[i,:] = (self.mu[N:] + (1 + self.alpha) * self.SigT[i] * self.binWidth / 2)[:]
      # right to left
      self.c1rl[i,:] = (np.abs(self.mu[:N]) - (1 - np.abs(self.alpha)) * self.SigT[i] * self.binWidth / 2)[:]
      self.c2rl[i,:] = (np.abs(self.mu[:N]) + (1 + np.abs(self.alpha)) * self.SigT[i] * self.binWidth / 2)[:]

    while(self.currentEps > self.epsilon):
      iterationNum += 1
      # run a transport sweep
      oldScalarFlux = np.copy( self.scalarFlux[:] )
      self.transportSweep()

      if self.diagnostic == True and iterationNum > 1:
        # calculate new rho estimate
        self.estimateRho(oldScalarFlux)
        # calculate new epsilon to test convergence
        self.testConvergence(oldScalarFlux)
        # call writeOutput
        self.writeOutput(self.out)
        # call the plotter
        self.plotScalarFlux(iterationNum)

      if iterationNum + 1 == self.maxIter:
        break

    # the simulation is done, write the scalar flux to the output file
    with open(self.out , "a") as output:
      output.write("\r\n Scalar Flux: \r\n")
      for i , val in enumerate(self.scalarFlux):
        output.write('{:1.4f}'.format( i * self.width / self.numBins ) + "   " + '{:1.7f}'.format(val) )


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
      homogenous    =  booleanize( conf['Slab']['homogenous'].strip() )
      bins          =  int(        conf['Slab']['bins'].strip()       )
      width         =  float(      conf['Slab']['width'].strip()      )
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
  # hardcoded values: 100 maximum iterations, alpha=0 for diamond difference
  slab = Slab(bins , width , loud=loud , diagnostic=diagnostic , quadSetOrder=quadSetOrder , epsilon=epsilon , out=outputFi , maxIter=100 , alpha=0)
  if homogenous == True:
    slab.setHomogenousData(SigT , SigS , Q)
  else:
    raise NotImplementedError

  slab.setRightBoundaryFlux( rightFlux , boundaryType = right)
  slab.setLeftBoundaryFlux(  leftFlux , boundaryType = left )
  slab.run()


