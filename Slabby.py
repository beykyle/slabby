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

# -------------------------------------------------------------------------------------- #
#
#  Thomas Algorithm for tridiagonal matrix inversion
#
# -------------------------------------------------------------------------------------- #

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
  nf = len(a)     # number of equations
  ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
  for it in range(1, nf):
    mc = ac[it]/bc[it-1]
    bc[it] = bc[it] - mc*cc[it-1]
    dc[it] = dc[it] - mc*dc[it-1]

  xc = ac
  xc[-1] = dc[-1]/bc[-1]

  for il in range(nf-2, -1, -1):
    xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

  del bc, cc, dc  # delete variables from memory

  return( xc )


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

      data = dat.readlines()[1:]

    self.numRegions  = len(data)
    self.Q        = np.zeros(self.numRegions)
    self.SigT     = np.zeros(self.numRegions)
    self.sigs     = np.zeros(self.numregions)
    self.z        = np.zeros(self.numregions)

    for i , line in enumerate(data):
      line  = [x.strip().rstrip("\n\r") for x in  line.split(",")]
      self.Q[i]    = float(line[qInd])
      self.SigT[i] = float(line[tInd])
      self.SigS[i] = float(line[sInd])
      self.z[i]    = float(line[zInd])

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
      self.z = np.linspace(zmin , zmax , num=(numBins+1))

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
    return(len(self.z) - 1)

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
    for i in range(0,len(self.mesh.z)-1):
      # find z in the middle of the mesh bin
      z = self.mesh.z[i] + self.mesh.binWidth(i)*0.5
      dist = np.finfo(z.dtype).max
      for j in range(0,len(material.z)-1):
        newdist = np.fabs(material.z[j] - z)
        if newdist < dist:
          dist = newdist
          matchIndex = j

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

  def reg2coarse(self , i):
    # only use if acceleration == 'CMFD'
    # assuming regular bin sizes in both coarse and regular mesh TODO
    return( i // ( self.mesh.numBins() // self.coarseMesh.numBins()  ))

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

  def getCellEdgeCurrent(self , psi_right , *args , **kwargs ):
    # apply the 1st order angular moment avergaing operator to the
    # angular flux in a spatial bin according to the Gauss-Legendre quad sets
    offset = 0
    if kwargs is not None:
      for key, value in kwargs.items():
        if key == "direction" and value == "right":
          offset = int(self.quadSetOrder / 2)
    phi1 = 0
    for i , val in enumerate(psi_right):
      phi1 += self.mu[i + offset] * val * self.weights[i + offset]
    return(phi1)

  def getScatterSource(self):
    # for an isotropically scattering problem we dont need to calculate any
    # angular moments of the scalar flux
    return( 0.5 * ( np.multiply(self.SigS , self.scalarFlux) + self.Q ) )

  def getCoarseEdgeCurrent(self):
    fineBin = 0
    coarseEdgeCurrent = np.zeros( self.coarseMesh.numBins() )
    for i in range( self.coarseMesh.numBins() - 1):
      fineBin += int( self.coarseMesh.map[i] - 1 )
      coarseEdgeCurrent[i] = self.edgeCurrent[fineBin]

    return( coarseEdgeCurrent )

  def cellCenteredDiffusion(self):
    # calculate cell edge total cross section and cell edge binWidth
    if self.iterationNum == 2:
      self.cellEdgeBinWidth = np.ones( self.mesh.numBins() )
      self.cellEdgeSigT = np.ones( self.mesh.numBins() )
      self.cellEdgeSigT[0]  = self.SigT[0]
      self.cellEdgeSigT[-1] = self.SigT[-1]
      self.cellEdgeBinWidth[0]  = 0.5 * self.mesh.binWidth(0)
      self.cellEdgeBinWidth[-1] = 0.5 * self.mesh.binWidth( self.mesh.numBins() -1 )
      for i in range(0, self.mesh.numBins() -1):
        self.cellEdgeSigT[i] = ( self.mesh.binWidth(i) * self.SigT[i] + self.mesh.binWidth(i+1) * self.SigT[i+1] ) / \
                               (self.mesh.binWidth(i) + self.mesh.binWidth(i+1))
        self.cellEdgeBinWidth[i] =  0.5 * (self.mesh.binWidth(i) +  self.mesh.binWidth(i+1))

    # calculate tridiagonal matrix elements
    A = np.zeros( self.mesh.numBins() ) # middle diagonal
    B = np.zeros( self.mesh.numBins() ) # outer diagonal
    S = np.zeros( self.mesh.numBins() ) # this is the b in Ax=b

    A[0]   =  1 / (3 * self.cellEdgeSigT[0] * self.cellEdgeBinWidth[0])     +  \
             (self.SigT[0] - self.SigS[0]) * self.mesh.binWidth(0)
    B[0]   = -1 / (3 * self.cellEdgeSigT[0] * self.cellEdgeBinWidth[0])
    B[-1]  = -1 / (3 * self.cellEdgeSigT[-1] * self.cellEdgeBinWidth[-1])
    S[0]   =  self.Q[0] * self.mesh.binWidth(0)

    for i in range( 1 ,  self.mesh.numBins() - 1 ):
      A[i] = 1 / (3 * self.cellEdgeSigT[i] * self.cellEdgeBinWidth[i])     +  \
             1 / (3 * self.cellEdgeSigT[i-1] * self.cellEdgeBinWidth[i-1]) +  \
             (self.SigT[i] - self.SigS[i]) * self.mesh.binWidth(i)
      B[i] = -1 / (3 * self.cellEdgeSigT[i] * self.cellEdgeBinWidth[i])
      S[i] = self.Q[i] * self.mesh.binWidth(i)

    return( TDMAsolver(A , B , B , S ) )

  def coarseDiffusion(self ):
    # set the flux weighted and volume weighted scalar fluxes and cross sections
    coarseScalarFlux  =            self.getVolumeAvgToCoarseMesh( self.scalarFlux )
    coarseSigT        = np.divide( self.getVolumeAvgToCoarseMesh( np.multiply( self.scalarFlux , self.SigT) )  ,
                        coarseScalarFlux )
    coarseSigA        = np.divide( self.getVolumeAvgToCoarseMesh( np.multiply( self.scalarFlux , self.SigT - self.SigS) )  ,
                        coarseScalarFlux )

    # calculate coarse cell edge curreny\t
    coarseEdgeCurrent = self.getCoarseEdgeCurrent()

    # determine Dk for each interior coarse cell edge
    D = np.zeros( self.coarseMesh.numBins() - 1)
    for i in range(0, self.coarseMesh.numBins() - 1):
      D[i] = (coarseEdgeCurrent[i] +
               2/3 * ( ( coarseScalarFlux[i+1] - coarseScalarFlux[i]  ) /
                       (coarseSigT[i+1] * self.coarseMesh.binWidth(i+1)  +
                         coarseSigT[i] * self.coarseMesh.binWidth(i) ) )
              ) / ( coarseScalarFlux[i+1] - coarseScalarFlux[i]  )

    # determine updated coarse grid scalar flux
    # create symmetric tridiagonal system: A and B:  [ A1 -B1         ... 0 ...   ]
    #                                                [-B1  A2 -B2     ... 0 ...   ]
    #                                                [-B2  A3 -B3     ... 0 ...   ]
    #                                                [ 0  -B3  A4 -B4 ... 0 ...   ]
    #                                                [ .       .   .   .          ]
    #                                                [ .           .   .   .      ]
    #                                                [ .               .   .   .  ]

    A = np.ones( self.coarseMesh.numBins() ) # middle diagonal
    B = np.ones( self.coarseMesh.numBins() ) # outer diagonal
    S = np.ones( self.coarseMesh.numBins() ) # this is the b in Ax=b

    A[0] =  - coarseSigA[0] * self.coarseMesh.binWidth(0)                                                                +  \
            - (2/3) * ( 1/(coarseSigT[1] * self.coarseMesh.binWidth(1) + coarseSigT[0] * self.coarseMesh.binWidth(0) )   +  \
                        1/(coarseSigT[1] * self.coarseMesh.binWidth(1) + coarseSigT[0] * self.coarseMesh.binWidth(0) ) ) +  \
              self.getCellEdgeCurrent( self.leftBoundaryFlux  , direction="right") + D[1]

    A[-1] = - coarseSigA[-1] * self.coarseMesh.binWidth(-1)                                                               + \
            - (2/3) * ( 1/(coarseSigT[-1] * self.coarseMesh.binWidth( self.coarseMesh.numBins() - 1)                      + \
                           coarseSigT[-2] * self.coarseMesh.binWidth( self.coarseMesh.numBins() - 1 ) )                   + \
                      ( 1/(coarseSigT[-1] * self.coarseMesh.binWidth( self.coarseMesh.numBins() - 1)                      + \
                           coarseSigT[-2] * self.coarseMesh.binWidth( self.coarseMesh.numBins() - 2 ) ) ) )               + \
              self.getCellEdgeCurrent( self.leftBoundaryFlux  , direction="right") + D[1]

    # integrate left boundary flux and determine B0
    B[0]  = (2/3) * ( 1/(coarseSigT[1] * self.coarseMesh.binWidth(1) + coarseSigT[0] * self.coarseMesh.binWidth(0) ) ) + \
            self.getCellEdgeCurrent( self.leftBoundaryFlux  , direction="right")

    # integrate right boundary flux and determine BK
    B[-1]  = (2/3) * ( 1/(coarseSigT[-1] * self.coarseMesh.binWidth( self.coarseMesh.numBins() - 1 )     + \
                          coarseSigT[-2] * self.coarseMesh.binWidth( self.coarseMesh.numBins() - 2 ) ) ) + \
             self.getCellEdgeCurrent( self.leftBoundaryFlux  , direction="right")

    for i in range( 1 ,  self.coarseMesh.numBins() - 1 ):
      A[i] =   - coarseSigA[i] * self.coarseMesh.binWidth(i)                                                                    +  \
               - (2/3) * ( 1/(coarseSigT[i+1] * self.coarseMesh.binWidth(i+1) + coarseSigT[i] * self.coarseMesh.binWidth(i) )   +  \
                           1/(coarseSigT[i-1] * self.coarseMesh.binWidth(i-1) + coarseSigT[i] * self.coarseMesh.binWidth(i) ) ) +  \
               D[i-1] + D[i]
      B[i] =   (2/3) * ( 1/(coarseSigT[i+1] * self.coarseMesh.binWidth(i+1) + coarseSigT[i] * self.coarseMesh.binWidth(i) ) )   + D[i]
      S[i] =   self.coarseQ[i] * self.coarseMesh.binWidth(i)

    return( TDMAsolver(A , B , B , S ) )

  def getVolumeAvgToCoarseMesh( self , vec ):
    # given a vector over the regular mesh, returns a vector over the coarse mesh
    coarseVec = np.zeros( self.coarseMesh.numBins() )

    for (i , z) in enumerate( self.mesh.z[:-1] ):
      coarseVec[ self.reg2coarse(i) ] += vec[i] * self.mesh.binWidth(i)

    for (j) in range(0, self.coarseMesh.numBins() ):
      coarseVec[j] = coarseVec[j] / self.coarseMesh.binWidth(j)

    return( coarseVec )

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
      psiOut = np.divide( (np.multiply( self.c1lr[i,:] , psiIn[:]  ) + Sn[i] *
                          self.mesh.binWidth(i) ) , self.c2lr[i,:]
                        )
      # get the cell edge current if needed for acceleration
      if (self.acceleration == "CMFD" ):
        self.edgeCurrent[i] += self.getCellEdgeCurrent( psiOut , direction="right")
      # find the average flux in the bin according to the spatial differencing scheme
      psiAv  = (1 + self.alpha[i][ :int( self.quadSetOrder / 2) ]) * 0.5 * psiOut[:] +\
               (1 - self.alpha[i][ :int( self.quadSetOrder / 2) ]) * 0.5 * psiIn[:]
      # find the scalar flux in this spatial bin from right-moving flux
      self.scalarFlux[i] = self.getScalarFlux(psiAv , direction="right")
      # set the incident flux on the next bin to exiting flux from this bin
      psiIn = psiOut

    # find the right boundary flux
    if (self.rightBoundaryType == "reflecting"):
      psiIn = self.rightBoundaryFlux + psiOut[::-1] # reverse psiOut array
    else:
      psiIn = self.rightBoundaryFlux

    # initialize variables used in the sweep
    psiAv  = np.zeros(( int( self.quadSetOrder / 2)))
    psiOut = np.zeros(( int( self.quadSetOrder / 2)))

    # sweep right to left
    for i in range(self.numBins - 1 , -1 , -1):
      # find the upstream flux at the left bin boundary
      psiOut = np.divide( (np.multiply( self.c1rl[i,:] , psiIn[:]  ) + Sn[i] *
                          self.mesh.binWidth(i) ) , self.c2rl[i,:]
                        )
      # get the cell edge current if needed for acceleration
      if (self.acceleration == "CMFD" ):
        self.edgeCurrent[i] += self.getCellEdgeCurrent( psiOut , direction="left")
      # find the average flux in the bin according to the spatial differencing scheme
      psiAv  = (1 + np.abs(self.alpha[i][ int( self.quadSetOrder / 2): ])) * 0.5 * psiOut[:] +\
               (1 - np.abs(self.alpha[i][ int( self.quadSetOrder / 2): ])) * 0.5 * psiIn[:]
      # find the scalar flux in this spatial bin from left-moving flux
      self.scalarFlux[i] += self.getScalarFlux(psiAv , direction="left")
      # set the incident flux on the next bin to exiting flux from this bin
      psiIn = psiOut

    return(psiOut)

  def estimateRho(self , oldError):
    currentError = self.scalarFlux - self.oldScalarFlux
    rho = np.sqrt( np.dot(currentError , currentError ) / (np.dot(oldError , oldError ) ) )
    return(rho)

  def testConvergence(self , oldError):
    return( np.fabs(max( np.divide( oldError  , np.abs(self.scalarFlux) ) ) ))

  def clearOutput(self):
    with open(self.out , "w") as outt:
      outt.write("Running 1-D transport! \r\n")

  def run(self):
    """
    constants precomputed and vectorized for speed
    In a 3D code this would typically exceed memory limits of most devices, and be impractical
    but it allows for drastic speed improvements in the 1D case

    """
    # precompute alphas - constant matrix discretized over space and angle
    N = int(self.quadSetOrder / 2)
    if self.stepMethod == "diamond":
      self.alpha = np.zeros([self.numBins , len(self.mu)])

    if self.stepMethod == "implicit":
      self.alpha = np.ones([self.numBins , len(self.mu)])
      self.alpha[:][:N] = self.alpha[:][:N] * -1

    elif self.stepMethod == "characteristic":
      self.alpha =  np.ones([self.numBins , len(self.mu)])
      nom = np.zeros( len( self.SigT ))
      for i in range(0,len(nom)):
        nom = self.SigT[i] * self.mesh.binWidth(i)
        tau = nom / self.mu
        self.alpha[i][:] *=  1 / np.tanh(tau / 2)  - 2 / tau

    # inital scalar flux guess
    self.scalarFlux    = np.zeros(self.numBins)
    self.oldScalarFlux = np.zeros(self.numBins)
    self.iterationNum = 0
    self.clearOutput()

    # call the plotter
    if self.loud == True:
      self.plotScalarFlux(self.iterationNum)

    # precompute coefficients for solving for upstream SI
    # coefficients form a constant matrix, discretized over both angle and space
    self.c1lr = np.zeros((self.numBins , N ))
    self.c2lr = np.zeros((self.numBins , N ))
    self.c1rl = np.zeros((self.numBins , N ))
    self.c2rl = np.zeros((self.numBins , N ))

    if ( self.acceleration == "CMFD"):
      self.coarseQ     =  self.getVolumeAvgToCoarseMesh( self.Q    )
      self.coarseSigT  =  self.getVolumeAvgToCoarseMesh( self.SigT )
      self.coarseEdgeCurrent = np.zeros( self.coarseMesh.numBins() )
      self.edgeCurrent       = np.zeros( self.mesh.numBins() )

    for i in range(0 , self.numBins ):
      # left to right
      self.c1lr[i,:] = (self.mu[N:] - (1 - self.alpha[i][N:]) * self.SigT[i] *
                        self.mesh.binWidth(i) / 2
                       )[:]
      self.c2lr[i,:] = (self.mu[N:] + (1 + self.alpha[i][N:]) * self.SigT[i] *
                        self.mesh.binWidth(i) / 2
                       )[:]
      # right to left
      self.c1rl[i,:] = (np.abs(self.mu[:N]) - (1 - np.abs(self.alpha[i][:N])) * self.SigT[i] *
                        self.mesh.binWidth(i) / 2
                       )[:]
      self.c2rl[i,:] = (np.abs(self.mu[:N]) + (1 + np.abs(self.alpha[i][:N])) * self.SigT[i] *
                        self.mesh.binWidth(i) / 2
                       )[:]

    while(self.currentEps > self.epsilon):
      #self.plotScalarFlux(iterationNum)
      self.iterationNum += 1
      # run a transport sweep
      oldError = self.scalarFlux - self.oldScalarFlux
      self.oldScalarFlux = np.copy( self.scalarFlux[:] )
      psiOut = self.transportSweep()

      if (self.acceleration == "CMFD") and self.iterationNum > 1:
        self.oldCoarse = self.coarseDiffusion()

      if (self.acceleration == 'idsa' and self.iterationNum >= 2):
        self.scalarFlux = self.scalarFlux + self.cellCenteredDiffusion()

      if (self.acceleration == "CMFD") and self.iterationNum > 2:
        # calculate P1 approximation to iteration errors on the coarse mesh
        self.newCoarse = self.coarseDiffusion()
        # update each cell in the regular mesh
        for (i , z) in enumerate(self.mesh.z[:-1]):
          # find which coarse cell we're in
          j = self.reg2coarse(i)
          self.scalarFlux[i] = self.scalarFlux[i] * self.newCoarse[j] / self.oldCoarse[j]

        self.oldCoarse = np.copy( self.newCoarse[:] )

      if self.iterationNum > 1:
        # calculate new rho estimate
        self.rho = self.estimateRho(oldError)
        # calculate new epsilon to test convergence
        self.currentEps = self.testConvergence(oldError)

      if self.diagnostic == True and self.iterationNum > 1:
        # call writeOutput
        self.writeOutput(self.out)

      # call the plotter
      if self.loud == True:
        self.plotScalarFlux(self.iterationNum)

      if self.iterationNum + 1 == self.maxIter:
        break

    print( "Converged in " + str(self.iterationNum) + " iterations" )
    # the simulation is done, write the scalar flux to the output file
    if (self.transmission == True):
      #print( "right: " + '{:3.4E}'.format( self.getCellEdgeCurrent( psiOut , direction="right" )  ) )
      #print( "left: "  + '{:3.4E}'.format( self.getCellEdgeCurrent( self.leftBoundaryFlux , direction="right" ) ) )
      print( "Transmission probability: " + '{:3.4E}'.format( self.getCellEdgeCurrent( psiOut , direction="right" )  /
                                                              self.getCellEdgeCurrent( self.leftBoundaryFlux , direction="right" ) ) )
    with open(self.out , "a") as output:
      output.write("\r\n x , Scalar Flux: \r\n")
      for i , val in enumerate(self.scalarFlux):
        if self.LaTeX == True:
          output.write('{:1.4f}'.format( i * self.width / self.numBins ) +
                       " & " + '{:1.7f}'.format(val) + r"\\" + " \r\n")
        else:
          output.write('{:1.4f}'.format( i * self.width / self.numBins ) +
                      " , " + '{:1.7f}'.format(val) + "\r\n")

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

  # run the slab
  slab.run()

