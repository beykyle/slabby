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
__email__ = "beykyle@umich.edu"
__status__ = "Development"


# -------------------------------------------------------------------------------------- #
#
#  Material class
#
# -------------------------------------------------------------------------------------- #

class Material:
  def __init__(self , Q=0 , SigT=0 , SigS=0 , z=[0,10] , matData=None):
    """
    Each input is a vector corresponding to the material data in a region of the problem
    z is a vector giving the left edge of each region, and the final right edge
    """
    if matData != None:
      self.getMatDataFromFile(matData)
    else:
      self.Q    = Q
      self.SigT = SigT
      self.SigS = SigS
      self.z    = z

  def getMatDataFromFile(self , filename):
    with open(filename, "r") as dat:
      headers    = dat.readline()
      for i , header in enumerate(headers.split(",")):
        if   header.strip().rstrip("\r\n") == "z":
          zInd = i
        elif   header.strip().rstrip("\r\n") == "SigS":
          sInd = i
        elif header.strip().rstrip("\r\n") == "SigT":
          tInd = i
        elif header.strip().rstrip("\r\n") == "Q":
          qInd = i

      data = dat.readlines()

    self.numRegions  = len(data) - 1
    self.Q        = np.zeros(self.numRegions)
    self.SigT     = np.zeros(self.numRegions)
    self.SigS     = np.zeros(self.numRegions)
    self.z        = np.zeros(self.numRegions+1)

    for i , line in enumerate(data):
      line  = [x.strip().rstrip("\n\r") for x in  line.split(",")]
      self.z[i]    = float(line[zInd])
      if i < len(data)-1:
        self.Q[i]    = float(line[qInd])
        self.SigT[i] = float(line[tInd])
        self.SigS[i] = float(line[sInd])

# -------------------------------------------------------------------------------------- #
#
#  Mesh class
#
# -------------------------------------------------------------------------------------- #

class Mesh:

  def __init__(self , zmin=0 , zmax=10 , numBins=100 , inputFile=None):
    if inputFile == None:
      """
      zmin corresponds to the left edge of the first bin
      zmax corresponds to the right edge of the last bin
      """
      self.z = np.linspace(zmin , zmax , num=(numBins) + 1 )

    else:
      """
      If an inputFIle kwarg is passed, all other parameters will be ignored
      Input file must have, as the 1st column, the z values of every left bin edge,
      as well as the final right bin edge, in cm
      """
      self.setupFromInputFile(inputFile)

  def setupFromInputFile(self , inputfile):
    raise NotImplementedError

  def binWidth(self , index):
    """
    indexing is from 0
    """
    return( self.z[index + 1] - self.z[index])

  def totalWidth(self):
    return(self.z[-1] - self.z[0])

  def numBins(self):
    return( len(self.z) - 1)

# -------------------------------------------------------------------------------------- #
#
#  Slab class
#
# -------------------------------------------------------------------------------------- #

class Slab:
  def __init__(self , mesh , materialData , method="SI" , stepMethod='diamond', loud=False ,
               acceleration='none', relaxationFactor = 1 , diagnostic=False , LaTeX=False ,
               epsilon=1E-8, quadSetOrder=8  , out="Slab.out" , maxIter=10000 , transmission=True):
    """
    Mandatory inputs are a mesh object and material object

    """

    if loud == True:
      print("Intializing slab!")

    # initialize general parameters
    self.loud  = loud
    self.diagnostic = diagnostic
    self.LaTeX = LaTeX
    self.out = out
    self.maxIter = maxIter

    # set numerical settings
    self.relaxationFactor = relaxationFactor
    self.epsilon = epsilon
    self.quadSetOrder = quadSetOrder
    self.setDifferencingScheme(stepMethod)
    self.setAccelerationMethod(acceleration)
    self.transmission = transmission

    # initialize slab material
    self.mesh = mesh
    self.initializeSlab( materialData)

    # intialize diagnostic parameters
    self.currentEps  = 1000 # a big number
    self.rho         = 0

    self.epsilons = []
    self.rhos     = []
    self.its      = []

    # initialize quad set weights ,  make sure they're normalized
    self.mu , self.weights = roots_legendre(self.quadSetOrder , mu=False)
    self.weights = np.array(self.weights)
    self.weights = self.weights / sum(self.weights)


    # initialize plotting axes
    if self.loud == True:
      self.initializeFigure()

  # ------------------------------------------------------------------------------------ #
  #
  #  Slab class -- slab and numerical method setup
  #
  # ------------------------------------------------------------------------------------ #

  def initializeSlab(self , material ):
    self.numBins  = self.mesh.numBins()
    self.width    =  self.mesh.totalWidth()
    self.Q , self.SigT , self.SigS = self.interpolateMaterialToMesh(material)

    print("number of bins: " + str(self.numBins))
    print("slab width : "    + str(self.width))

  def interpolateMaterialToMesh(self , material):
    Q , SigT , SigS = np.zeros(self.numBins), np.zeros(self.numBins), np.zeros(self.numBins)
    matchIndex = 0
    for i in range(0,len(self.mesh.z)-1):
      for j in range(0, len(material.z) - 1):
        if (  self.mesh.z[i+1] < material.z[j+1] and self.mesh.z[i] > material.z[j]  ):
          matchIndex = j
        elif( self.mesh.z[i+1] - material.z[j+1] >  self.mesh.z[i] -  material.z[j] and
            self.mesh.z[i] > material.z[j]      ):
          matchIndex = j + 1

      Q[i]    = material.Q[matchIndex]
      SigT[i] = material.SigT[matchIndex]
      SigS[i] = material.SigS[matchIndex]

    return( Q , SigT , SigS)

  def  setDifferencingScheme(self , stepMethod):
    # determine spatial finite differencing method
    if stepMethod == "implicit" or stepMethod == "diamond" or stepMethod == "characteristic":
      self.stepMethod  = stepMethod
    else:
      print("Unrecognized step method! Exiting.")
      sys.exit()
    print("Differencing scheme: " + self.stepMethod)

  def setAccelerationMethod(self , acceleration):
    # determine acceleration method
    if acceleration == 'CMFD' or acceleration == 'none' or acceleration == 'idsa':
      self.acceleration = acceleration
    else:
      raise NotImplementedError
    print("Acceleration method: " + self.acceleration)

  def addCoarseMesh(self , coarseMesh , method='CMFD'):
    self.setAccelerationMethod(method)
    self.coarseMesh   = True
    self.coarseMesh   = coarseMesh
    self.coarseMesh.map = np.zeros( self.mesh.numBins() )
    # find the number of fine mesh elements for each coarse mesh element
    for (i , z)  in enumerate(self.mesh.z ):
      self.coarseMesh.map[ self.reg2coarse(i) ] += 1

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
        self.rightBoundaryFlux[i] = rightFlux *  self.weights[i]
    elif self.rightBoundaryType == "vacuum":
      pass
    elif self.rightBoundaryType == "reflecting":
      pass
    else:
      print("Invalid boundary type: " + self.rightBoundaryType + " for right boundary! \r\n")
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
        self.leftBoundaryFlux[i] = leftFlux * self.weights[i]
    elif self.leftBoundaryType == "vacuum":
      pass
    else:
      print("Invalid boundary type: " + self.leftBoundaryType + " for left boundary! \r\n")
      sys.exit()

  # ------------------------------------------------------------------------------------ #
  #
  #  Slab class -- plotting and output
  #
  # ------------------------------------------------------------------------------------ #

  def writeOutput(self , filename):
    # write the diagnostics to the output file
    with open(filename , "a") as output:
      output.write("Epsilon: " + '{:1.9E}'.format(self.currentEps) + " , Rho: " +
          '{:1.9E}'.format(self.rho) + "\r\n")
    # print the diagnostics to the command line
    print("Epsilon: " + '{:1.9E}'.format(self.currentEps) + " , Rho: " + '{:1.9E}'.format(self.rho))

  def  initializeFigure(self):
    self.fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    self.font = { 'family': 'serif',
                  'color':  'black',
                  'weight': 'regular',
                  'size': 12,
                  }

    self.title_font = { 'family': 'serif',
                        'color':  'black',
                        'weight': 'bold',
                        'size': 12,
                      }

    self.ax1 = self.fig.add_subplot(grid[0,:])
    self.ax2 = self.fig.add_subplot(grid[1,0])
    self.ax3 = self.fig.add_subplot(grid[1,1])
    self.ax2.set_ylabel(r"Convergence Criterion, $\epsilon$" , fontdict=self.font)
    self.ax2.set_xlabel("Iteration Number" , fontdict=self.font)
    self.ax1.set_xlabel(r"$x$ [cm]", fontdict=self.font)
    self.ax1.set_ylabel(r"scalar flux, $\Phi$ [cm$^{-2}$ s$^{-1}$ ]", fontdict=self.font)

  def plotScalarFlux(self, iterNum ):
    plt.ion()
    plt.cla()
    self.ax1.clear()
    self.ax3.set_ylabel(r"Estimated ROC, $\rho$"             , fontdict=self.font)
    self.ax3.set_xlabel("Iteration Number" , fontdict=self.font)

    self.its.append(iterNum)
    if iterNum > 2:
      self.epsilons.append(self.currentEps)
      self.rhos.append(self.rho)
    else:
      self.epsilons.append(None)
      self.rhos.append(None)

    xint = range(0, iterNum+2 , int(round(iterNum / 10))+1 )
    self.ax3.set_xticks(xint)
    self.ax2.set_xticks(xint)
    self.ax2.semilogy(self.its , self.epsilons , 'r.' , label=r"$\epsilon$")
    self.ax2.semilogy([0,iterNum+1] , [self.epsilon , self.epsilon] , 'k--' , label="criterion")
    self.ax3.plot(self.its , self.rhos     , 'b.' , label=r"$\rho$")

    if iterNum == 0:
        self.ax2.legend()
        self.ax3.legend()

    x = np.linspace(0 , self.width , self.numBins)
    self.ax1.scatter(x , self.scalarFlux  , c='k' , marker='.')
    self.ax1.set_title("Iteration " + str(iterNum), fontdict=self.title_font)
    self.ax1.plot( [x[0]  , x[0]  ] , [0 , max(self.scalarFlux)*1.2 ]  , '--r'  )
    self.ax1.plot( [x[-1] , x[-1] ] , [0 , max(self.scalarFlux)*1.2 ]  , '--r' )
    adjustment = (x[-1] - x[0])*0.05
    self.ax1.text( x[0]  - adjustment     , 1.1*self.ax1.get_ylim()[1] ,
                  self.leftBoundaryType  , fontdict=self.font)
    self.ax1.text( x[-1] - 1.2*adjustment , 1.1*self.ax1.get_ylim()[1] ,
                  self.rightBoundaryType , fontdict=self.font)

    if ((self.loud == True) and
       ( self.currentEps >= self.epsilon or self.iterationNum > self.maxIter)):
      plt.draw()
      plt.pause(0.001)
    else:
      plt.ioff()
      plt.draw()
      input("Press ENTER to finish")
      print("finished!")

    # save converged figure
    if (self.currentEps >= self.epsilon or self.iterationNum > self.maxIter):
      plt.savefig("./flux_" + str(iterNum) + ".png")


  # ------------------------------------------------------------------------------------ #
  #
  #  Slab class -- numerical solve
  #
  # ------------------------------------------------------------------------------------ #

  def test(self):
    print( self.mesh )


# -------------------------------------------------------------------------------------- #
#
#  input parsing functions
#
# -------------------------------------------------------------------------------------- #

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

def getMaterialFromInput(conf):

  # read material section of input file
  if 'Material' in conf:
    homogenous    =  booleanize( conf['Material']['homogenous'].strip() )
    if 'matData' in conf['Material'] and homogenous == False:
      matData = conf['Material']['matData']
      material = Material(matData=matData)
    elif homogenous == True:
      SigT          =  float(      conf['Material']['SigT'].strip()       )
      SigS          =  float(      conf['Material']['SigS'].strip()       )
      Q             =  float(      conf['Material']['Q'].strip()          )
      width         =  float(      conf['Mesh']['width'].strip()      )
      material = Material( Q=np.array([Q]) ,
                           SigT=np.array([SigT]) ,
                           SigS=np.array([SigS]) ,
                           z=np.array([0,width])
                         )
    else:
      print("material data not inputted!")
      sys.exit()
  else:
    raise
    print("No Material section found in input file! Exiting! ")
    sys.exit()

  return(material)

def getMeshFromInput(conf):

  # read mesh section of input file
  if 'Mesh' in conf:
    regular    =  booleanize( conf['Mesh']['regular'].strip() )
    if 'meshData' in conf['Mesh'] and regular == False:
      meshData = conf['Mesh']['meshData']
      mesh = Mesh(inputFile=meshData)
    elif regular == True:
      bins          =  int(        conf['Mesh']['bins'].strip()       )
      width         =  float(      conf['Mesh']['width'].strip()      )
      mesh = Mesh(zmin=0 , zmax=width , numBins=bins)
    else:
      print("mesh data not inputted!")
      sys.exit()

  else:
    raise
    print("No Mesh section found in input file! Exiting! ")
    sys.exit()

  # create mesh object

  return(mesh)

def getCoarseMeshFromInput(inputfile):

  if "Coarse Mesh" in inputfile:
    mesh = Mesh(zmin=0 , zmax=float(  inputfile['Mesh']['width'].strip()       )  ,
                         numBins=int( inputfile['Coarse Mesh']['bins'].strip() )   )
  else:
    print("CMFD selected, but Coarse Mesh section not found!")
    sys.exit()

  return( mesh )

def getSlabFromInput(inputfile):
  conf = configparser.ConfigParser()
  conf.read(inputFile)

  if 'General' in conf:
    loud          =  booleanize( conf['General']['Loud'].strip()        )
    LaTeX         =  booleanize( conf['General']['LaTeX'].strip()        )
    diagnostic    =  booleanize( conf['General']['Diagnostic'].strip()  )
    outputFi      =              conf['General']['Output'].strip()
  else:
    print("No General section found in input file! Exiting! ")
    sys.exit()

  if 'Numerical Settings' in conf:
    stepMethod    =              conf['Numerical Settings']['stepMethod'].strip()
    DSA           =              conf['Numerical Settings']['acceleration'].strip()
    epsilon       =  float(      conf['Numerical Settings']['convergence'].strip()  )
    quadSetOrder  =  int(        conf['Numerical Settings']['quadSetOrder'].strip() )
    if quadSetOrder % 2 != 0:
      print("Order of the quad set must be even! Exiting!")
      sys.exit()
  else:
    print("No Numerical Settings section found in input file! Exiting! ")
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

  # get mesh data
  mesh = getMeshFromInput(conf)

  # get materilal data
  material = getMaterialFromInput(conf)

  # Create slab object and run the simulation
  # hardcoded values: 100 maximum iterations, alpha=0 for diamond difference
  slab = Slab(mesh , material , loud=loud , diagnostic=diagnostic , quadSetOrder=quadSetOrder ,
              epsilon=epsilon , out=outputFi , stepMethod=stepMethod , acceleration=DSA
              )


  slab.setRightBoundaryFlux( rightFlux , boundaryType=right)
  slab.setLeftBoundaryFlux(  leftFlux  , boundaryType=left )

  if (DSA == "CMFD"):
    slab.addCoarseMesh( getCoarseMeshFromInput( conf ) )

  return(slab)

# -------------------------------------------------------------------------------------- #
#
#  main
#
# -------------------------------------------------------------------------------------- #

if __name__ == '__main__':

  # if called from the command line, input file is the 1st command line arg
  inputFile = sys.argv[1]

  # set up slab
  slab = getSlabFromInput(inputFile)

  slab.test()

  # run the slab
  #t , r , n  = slab.run()

