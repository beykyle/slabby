#!/usr/bin/env py36


"""
Slabby is a Diamond-Differenced, discrete ordinates, 1-D planar geometry, fixed-source, monoenergetic, isotropic scattering neutron transport code

"""

import numpy as np
import configparser
import sys
from matplotlib.ticker import MaxNLocator
from scipy.special import roots_legendre
from matplotlib    import pyplot as plt

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Slab:
  def __init__(self , DSA="none" , LaTeX="False" , method="SI" , stepMethod='diamond', *args , **kwargs):
    print("Intializing slab!")
    self.LaTeX = LaTeX

    # determine method , DSA method, and set up differencing scheme
    if method == "SI":
      self.method  = method
      print("Method: " + self.method)
    else:
      print("Unrecognized method! Exiting.")
      sys.exit()

    if DSA == "none" or DSA == "CMFD":
      self.DSA  = DSA
      print("DSA method: " + self.DSA)
    else:
      print("Unrecognized DSA method! Exiting.")
      sys.exit()

    if stepMethod == "diamond":
      self.stepMethod  = stepMethod

    elif stepMethod == "characteristic":
      self.stepMethod  = stepMethod

    else:
      print("Unrecognized step method! Exiting.")
      sys.exit()
    print("Differencing scheme: " + self.stepMethod)

    # get key word constructor arguments
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "loud" or key == "diagnostic" or key == "homogenous" or key == "LaTeX":
          print(key + ": " + str(value))
          setattr(self , key , booleanize(value))
        elif key == "epsilon":
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
    self.currentEps  = 1000 # a big number
    self.rho         = 0

    self.epsilons = []
    self.rhos     = []
    self.its      = []

    # initialize quad set weights ,  make sure they're normalized
    self.mu , self.weights = roots_legendre(self.quadSetOrder , mu=False)
    self.weights = np.array(self.weights)
    self.weights = self.weights / sum(self.weights)


    if self.loud == True:
      self.fig = plt.figure(figsize=(12, 6))
      grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

      self.ax1 = self.fig.add_subplot(grid[0,:])
      self.ax2 = self.fig.add_subplot(grid[1,0])
      self.ax3 = self.fig.add_subplot(grid[1,1])

  def setHomogenousData(self , numBins , width , sigt , sigs , q):
    self.numBins     = int(numBins)
    self.binWidth    = float(width / numBins)
    self.width      =  float(width)
    self.Q           = np.zeros(numBins)
    self.SigT        = np.zeros(numBins)
    self.SigS        = np.zeros(numBins)
    for i in range(0,self.numBins):
      self.Q[i]    = q
      self.SigS[i] = sigs
      self.SigT[i] = sigt

    print("number of bins: " + str(self.numBins))
    print("slab width : "    + str(self.width))


  def getMatDataFromFile(self , filename):
    with open(filename, "r") as dat:
      self.width = float(dat.readline())
      headers    = dat.readline()
      for i , header in enumerate(headers.split(",")):
        if   header.strip().rstrip("\r\n") == "SigS":
          sInd = i
        elif header.strip().rstrip("\r\n") == "SigT":
          tInd = i
        elif header.strip().rstrip("\r\n") == "Q":
          qInd = i

      data = dat.readlines()[1:]

    self.numBins  = len(data)
    self.binWidth = float(self.width / self.numBins)
    self.Q        = np.zeros(self.numBins)
    self.SigT     = np.zeros(self.numBins)
    self.SigS     = np.zeros(self.numBins)

    for i , line in enumerate(data):
      line  = [x.strip().rstrip("\n\r") for x in  line.split(",")]
      self.Q[i]    = float(line[qInd])
      self.SigT[i] = float(line[tInd])
      self.SigS[i] = float(line[sInd])

    print("number of bins: " + str(self.numBins))
    print("slab width : "    + str(self.width))

  def writeOutput(self , filename):
    # write the diagnostics to the output file
    with open(filename , "a") as output:
      output.write("Epsilon: " + '{:1.9E}'.format(self.currentEps) + " , Rho: " + '{:1.9E}'.format(self.rho) + "\r\n")
    # print the diagnostics to the command line
    print("Epsilon: " + '{:1.9E}'.format(self.currentEps) + " , Rho: " + '{:1.9E}'.format(self.rho))

  def plotScalarFlux(self, iterNum ):
    plt.ion()
    plt.cla()
    self.ax1.clear()

    self.its.append(iterNum)
    if iterNum > 2:
      self.epsilons.append(self.currentEps)
      self.rhos.append(self.rho)
    else:
      self.epsilons.append(None)
      self.rhos.append(None)

    self.ax2.set_xlabel("Iteration Number")
    self.ax3.set_xlabel("Iteration Number")
    xint = range(0, iterNum+2 , int(round(iterNum / 10))+1 )
    self.ax3.set_xticks(xint)
    self.ax2.set_xticks(xint)
    self.ax2.set_ylabel(r"Convergence Criterion, $\epsilon$")
    self.ax3.set_ylabel(r"Estimated ROC, $\rho$")
    #self.ax3.set_ylim(bottom=0)
    self.ax2.plot(self.its , self.epsilons , 'r.' , label=r"$\epsilon$")
    self.ax2.plot([0,iterNum+1] , [self.epsilon , self.epsilon] , 'k--' , label="criterion")
    self.ax2.legend()
    self.ax3.plot(self.its , self.rhos     , 'b.' , label=r"$\rho$")

    x = np.linspace(0 , self.width , self.numBins)
    self.ax1.plot(x , self.scalarFlux , "k.")
    self.ax1.set_xlabel(r"$x$ [cm]")
    self.ax1.set_ylabel(r"scalar flux, $\Phi$ [cm$^{-2}$ s$^{-1}$ ]")
    self.ax1.set_title("Iteration " + str(iterNum))

    if self.loud == True and self.currentEps >= self.epsilon:
      plt.draw()
      plt.pause(0.001)
    else:
      plt.ioff()
      plt.draw()
      a = input("Press ENTER to finish")
      print("finished!")

    if diagnostic == True:
      plt.savefig("./flux_" + str(iterNum) + ".png")

  def setRightBoundaryFlux(self , rightFlux , *args , **kwargs):
    self.rightFlux = rightFlux
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "boundaryType":
          setattr(self , "rightBoundaryType" , value)

    self.rightBoundaryFlux = np.zeros( int(self.quadSetOrder / 2)) # handled at sweep time
    if self.rightBoundaryType   == "planar":
      self.rightBoundaryFlux[-1] =  rightFlux
    elif self.rightBoundaryType == "isotropic":
      for i , val in enumerate(self.rightBoundaryFlux):
        self.rightBoundaryFlux[i] = rightFlux / (4 * np.pi) * self.weights[i]
    elif self.rightBoundaryType == "vacuum":
      pass
    elif self.rightBoundaryType == "reflecting":
      pass
    else:
      print("Invalid boundary type: " + self.leftBoundaryType + " for left boundary! \r\n")
      sys.exit()

  def setLeftBoundaryFlux(self , leftFlux , *args , **kwargs):
    self.leftFlux = leftFlux
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "boundaryType":
          setattr(self , "leftBoundaryType" , value)

    self.leftBoundaryFlux = np.zeros(int(self.quadSetOrder / 2))
    if self.leftBoundaryType   == "planar":
      self.leftBoundaryFlux[0] = leftFlux # set mu=0 direction to the given flux
    elif self.leftBoundaryType == "isotropic":
      for i , val in enumerate(self.leftBoundaryFlux):
        self.leftBoundaryFlux[i] = leftFlux / (4 * np.pi) * self.weights[i]
    elif self.leftBoundaryType == "vacuum":
      pass
    else:
      print("Invalid boundary type: " + self.leftBoundaryType + " for left boundary! \r\n")
      sys.exit()

  def getScalarFlux(self , psi , *args , **kwargs):
    # integrate the angular flux in a spatial bin according to the Gauss-Legendre quad sets
    offset = 0
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "direction" and value == "right":
          offset = int(self.quadSetOrder / 2)
    phi = 0
    # integrate psi over the quadrature set
    for i , val in enumerate(psi):
      phi += val * self.weights[i + offset]
    return(phi)

  def getScatterSource(self):
    # for an isotropically scattering problem we dont need to calculate any
    # angular moments of the scalar flux
    return( 0.5 * ( np.multiply(self.SigS , self.scalarFlux) + self.Q ) )

  def diffusion(self):
    scalarFlux = np.zeros(len( self.scalarFlux ))
    return(scalarFlux)

  def transportSweep(self):
    N = int(self.quadSetOrder / 2)
    # this transport function explicitly iterates through the spatial variable
    # and is vectorized in the angular variable for SIMD optimization
    # it uses the numpy library for vector operations

    # precompute scatter source for the isotropic problem (only discretized over space)
    Sn = self.getScatterSource()

    # find the left boundary flux
    psiIn = self.leftBoundaryFlux

    # initialize variables used in the sweep
    psiAv  = np.zeros(( int( self.quadSetOrder / 2)))
    psiOut = np.zeros(( int( self.quadSetOrder / 2)))

    # sweep left to right
    for i in range(0,self.numBins):
      # find the upstream flux at the right bin boundary
      psiOut = np.divide( (np.multiply( self.c1lr[i,:] , psiIn[:]  ) + Sn[i] *  self.binWidth ) , self.c2lr[i,:] )
      # find the average flux in the bin according to the spatial differencing scheme
      psiAv  = (1 + self.alpha[i][ :int( self.quadSetOrder / 2) ]) * 0.5 * psiOut[:] + \
               (1 - self.alpha[i][ :int( self.quadSetOrder / 2) ]) * 0.5 * psiIn[:]
      # find the scalar flux in this spatial bin from right-moving flux
      self.scalarFlux[i] = self.getScalarFlux(psiAv , direction="right")
      # set the incident flux on the next bin to exiting flux from this bin
      psiIn = psiOut

    # find the right boundary flux
    if (self.rightBoundaryType == "reflecting"):
      psiIn = self.rightBoundaryFlux + psiOut
    else:
      psiIn = self.rightBoundaryFlux

    # initialize variables used in the sweep
    psiAv  = np.zeros(( int( self.quadSetOrder / 2)))
    psiOut = np.zeros(( int( self.quadSetOrder / 2)))

    # sweep right to left
    for i in range(self.numBins -1 , 0 , -1):
      # find the upstream flux at the left bin boundary
      psiOut = np.divide( (np.multiply( self.c1rl[i,:] , psiIn[:]  ) + Sn[i] * self.binWidth ) , self.c2rl[i,:] )
      # find the average flux in the bin according to the spatial differencing scheme
      psiAv  = (1 + np.abs(self.alpha[i][ int( self.quadSetOrder / 2): ])) * 0.5 * psiOut[:] + \
               (1 - np.abs(self.alpha[i][ int( self.quadSetOrder / 2): ])) * 0.5 * psiIn[:]
      # find the scalar flux in this spatial bin from left-moving flux
      self.scalarFlux[i] += self.getScalarFlux(psiAv , direction="left")
      # set the incident flux on the next bin to exiting flux from this bin
      psiIn = psiOut

  def estimateRho(self , oldError):
    currentError = self.scalarFlux - self.oldScalarFlux
    rho = np.sqrt( np.dot(currentError , currentError ) / (np.dot(oldError , oldError ) ) )
    return(rho)

  def testConvergence(self , oldError):
    return( max( np.divide( oldError  , np.abs(self.scalarFlux) ) ) )

  def clearOutput(self):
    with open(self.out , "w") as outt:
      outt.write("Running 1-D transport! \r\n")

  def run(self):

    # precompute alphas
    if self.stepMethod == "diamond":
      self.alpha = np.zeros([self.numBins , len(self.mu)])

    elif self.stepMethod == "characteristic":
      self.alpha =  np.ones([self.numBins , len(self.mu)])
      nom = self.SigT * self.binWidth
      for i in range(0,len(nom)):
        tau = nom[i] / self.mu
        self.alpha[i][:] *=  1 / np.tanh(tau / 2)  - 2 / tau

    # inital scalar flux guess
    self.scalarFlux    = np.zeros(self.numBins)
    self.oldScalarFlux = np.zeros(self.numBins)
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
      self.c1lr[i,:] = (self.mu[N:] - (1 - self.alpha[i][N:]) * self.SigT[i] * self.binWidth / 2)[:]
      self.c2lr[i,:] = (self.mu[N:] + (1 + self.alpha[i][N:]) * self.SigT[i] * self.binWidth / 2)[:]
      # right to left
      self.c1rl[i,:] = (np.abs(self.mu[:N]) - (1 - np.abs(self.alpha[i][:N])) * self.SigT[i] * self.binWidth / 2)[:]
      self.c2rl[i,:] = (np.abs(self.mu[:N]) + (1 + np.abs(self.alpha[i][:N])) * self.SigT[i] * self.binWidth / 2)[:]

    while(self.currentEps > self.epsilon):
      #self.plotScalarFlux(iterationNum)
      iterationNum += 1
      # run a transport sweep
      oldError = self.scalarFlux - self.oldScalarFlux
      self.oldScalarFlux = np.copy( self.scalarFlux[:] )
      self.transportSweep()

      if self.diagnostic == True and iterationNum > 1:
        # calculate new rho estimate
        self.rho = self.estimateRho(oldError)
        # calculate new epsilon to test convergence
        self.currentEps = self.testConvergence(oldError)
        # call writeOutput
        self.writeOutput(self.out)
        # call the plotter
        if self.loud == True:
          self.plotScalarFlux(iterationNum)

      if iterationNum + 1 == self.maxIter:
        break

    # the simulation is done, write the scalar flux to the output file
    with open(self.out , "a") as output:
      output.write("\r\n x , Scalar Flux: \r\n")
      for i , val in enumerate(self.scalarFlux):
        if self.LaTeX == True:
          output.write('{:1.4f}'.format( i * self.width / self.numBins ) + " & " + '{:1.7f}'.format(val) + r"\\" + " \r\n")
        else:
          output.write('{:1.4f}'.format( i * self.width / self.numBins ) + " , " + '{:1.7f}'.format(val) + "\r\n")


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
    LaTeX         =  booleanize( conf['General']['LaTeX'].strip()        )
    diagnostic    =  booleanize( conf['General']['Diagnostic'].strip()  )
    outputFi      =              conf['General']['Output'].strip()
    method        =              conf['General']['Method'].strip()
    stepMethod    =              conf['General']['stepMethod'].strip()
    DSA           =              conf['General']['DSA'].strip()
    epsilon       =  float(      conf['General']['convergence'].strip()  )
    quadSetOrder  =  int(        conf['General']['quadSetOrder'].strip() )
    if quadSetOrder % 2 != 0:
      print("Order of the quad set must be even! Exiting!")
      sys.exit()

  else:
    print("No General section found in input file! Exiting! ")
    sys.exit()

  if 'Slab' in conf:
    homogenous    =  booleanize( conf['Slab']['homogenous'].strip() )
    if 'matData' in conf['Slab'] and homogenous == False:
      matData = conf['Slab']['matData']
    elif homogenous == True:
      bins          =  int(        conf['Slab']['bins'].strip()       )
      width         =  float(      conf['Slab']['width'].strip()      )
      SigT          =  float(      conf['Slab']['SigT'].strip()       )
      SigS          =  float(      conf['Slab']['SigS'].strip()       )
      Q             =  float(      conf['Slab']['Q'].strip()          )
    else:
      print("mat data not inputted!")
      sys.exit()
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
  slab = Slab(loud=loud       , method=method , diagnostic=diagnostic , quadSetOrder=quadSetOrder ,
              epsilon=epsilon , out=outputFi  , maxIter=100 , stepMethod=stepMethod , DSA=DSA
              )
  if homogenous == True:
    slab.setHomogenousData(bins , width , SigT , SigS , Q)
  else:
    slab.getMatDataFromFile(matData)

  slab.setRightBoundaryFlux( rightFlux , boundaryType=right)
  slab.setLeftBoundaryFlux(  leftFlux  , boundaryType=left )
  slab.run()


