#!/usr/bin/env py36


"""
Slabby is a Diamond-Differenced, discrete ordinates, 1-D planar geometry, fixed-source, monoenergetic, isotropic scattering neutron transport code

"""

import numpy as np
import configparser
import sys
from matplotlib import pyplot as plt

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

def booleanize(value):
    """Return value as a boolean."""
    true_values = ("yes", "true", "1")
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

  if '' in conf:
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



