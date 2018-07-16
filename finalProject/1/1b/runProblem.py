#!/usr/bin/env py36

"""
RunProblem.py is a wrapper for Slabby.py to efficiently run a series of problems with varying parameters

"""

import numpy as np
import configparser
import io
import sys
import contextlib
import configparser

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

sys.path.append("/home/kyle/Projects/umich/class/larsen_590/slabby/")

from Slabby import Slab , Mesh , Material

# -------------------------------------------------------------------------------------- #
#
#  context for supressing stdout
#
# -------------------------------------------------------------------------------------- #

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

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
      bins          =  [int(x.strip()) for x in conf['Mesh']['bins'].split(",") ]
      width         =  float(      conf['Mesh']['width'].strip()      )
    else:
      print("mesh data not inputted!")
      sys.exit()

  else:
    raise
    print("No Mesh section found in input file! Exiting! ")
    sys.exit()

  # create mesh object
  meshes = []
  for b in bins:
    meshes.append( Mesh(zmin=0 , zmax=width , numBins=b ) )

  return(meshes)

def getCoarseMeshFromInput(inputfile):

  if "Coarse Mesh" in inputfile:
    mesh = Mesh(zmin=0 , zmax=float(  inputfile['Mesh']['width'].strip()       )  ,
                         numBins=int( inputfile['Coarse Mesh']['bins'].strip() )   )
  else:
    print("CMFD selected, but Coarse Mesh section not found!")
    sys.exit()

  return( mesh )

def getInputFromFile(inputfile):
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
    stepMethod     =              conf['Numerical Settings']['stepMethod'].strip()
    DSA            =              conf['Numerical Settings']['acceleration'].strip()
    epsilon        =  float(      conf['Numerical Settings']['convergence'].strip()  )
    quadSetOrders  =  [int(x.strip()) for x in conf['Numerical Settings']['quadSetOrder'].split(",") ]
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
  meshes = getMeshFromInput(conf)

  # get materilal data
  material = getMaterialFromInput(conf)

  # Create slab object and run the simulation
  # hardcoded values: 100 maximum iterations, alpha=0 for diamond difference
  slabs = []
  settings = []

  for mesh in meshes:
    for quadSetOrder in quadSetOrders:
      settings.append( mesh )
      slabs.append(  Slab(mesh , material , loud=loud , diagnostic=diagnostic , quadSetOrder=quadSetOrder ,
                epsilon=epsilon , out=outputFi , stepMethod=stepMethod , acceleration=DSA ) )
      slabs[-1].setRightBoundaryFlux( rightFlux , boundaryType=right)
      slabs[-1].setLeftBoundaryFlux(  leftFlux  , boundaryType=left )
      if (DSA == "CMFD"):
        slabs[-1].addCoarseMesh( getCoarseMeshFromInput( conf ) )

  return(settings , slabs)

# -------------------------------------------------------------------------------------- #
#
#  Output writing
#
# -------------------------------------------------------------------------------------- #

def writeOutput(tr , rho , itera , settings):
  print( '{:1.3f}'.format( 30 / settings.numBins() ) + " & " +
       '{:1.6E}'.format(rho) + r"\\")

def writeTable(vec , form , side):
  # assuming assumptions
  string = side[0]
  for i in range(0,len(vec)):
    if (i+1)%6 != 0:
      string = string + form.format(vec[i]) + " & "
    else:
      string = string + form.format(vec[i]) + r"\\"
      print(string)
      if (i < len(vec) - 1):
        string = side[int((i+1)/6)]


# -------------------------------------------------------------------------------------- #
#
#  main
#
# -------------------------------------------------------------------------------------- #

if __name__ == '__main__':

  # if called from the command line, input file is the 1st command line arg
  inputFile = sys.argv[1]

  with nostdout():
    settings , slabs = getInputFromFile( inputFile )
  # run the slab
  tr  = []
  rh  = []
  ite = []
  for i , slab in enumerate(slabs):
    with nostdout():
      transmission , rho , itera = slab.run()
      tr.append(transmission)
      rh.append(rho)
      ite.append(itera)
    writeOutput( transmission , rho , itera  , settings[i] )

  print(rh)


  #print("\n")
  #writeTable(tr , '{:1.3E}' , ["2 & " , "4 & " , "8 & " ,"16 & " , "32 & "  ] )
  #writeTable(rh , '{:1.3E}' , ["2 & " , "4 & " , "8 & " ,"16 & " , "32 & "  ] )

