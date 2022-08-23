import sys
import os
from math import *
from numpy import arange, array, clip, exp, log

import json

def check_comment(comment_line, numelts):
    # make sure that we are on a comment line
    if not comment_line[0] == '!' and not (len(comment_line) == numelts):  
        print('wrong format for line:', comment_line)
        print('QUITTING')
        quit()


def check_num_values(values, numelts):
    # make sure that we have the expected number of values.
    if not (len(values) == numelts):
        print('wrong number of items for line:', values)
        print('QUITTING')
        quit()


def readXROTORFile(xrotorFileName):
    fid = open(xrotorFileName, 'r')

    if fid.readline().find('XROTOR') == -1:
        print('''This file does not look like an XROTOR input file, 
        it does not have XROTOR mentioned in its first line. Exiting.''')
        quit()
 
    # read in lines 2->8 which contains the run case information
    xrotorInputDict = {}
    line = fid.readline()
    comment_line = fid.readline().upper().split() 
    check_comment(comment_line, 5)

    values = fid.readline().split()
    check_num_values(values, 4)

    xrotorInputDict['vso'] = float(values[1])

    comment_line = fid.readline().upper().split()
    check_comment(comment_line, 5)
    values = fid.readline().split()
    check_num_values(values, 4)
    xrotorInputDict['rad'] = float(values[0])
    xrotorInputDict['vel'] = float(values[1])
    xrotorInputDict['adv'] = float(values[2])

    fid.readline()
    fid.readline()
    comment_line = fid.readline().upper().split()
    check_comment(comment_line, 2) 
    values = fid.readline().split()
    check_num_values(values, 1)

    nAeroSections = int(values[0])
    xrotorInputDict['nAeroSections'] = nAeroSections

    xrotorInputDict['rRstations'] = [0] * nAeroSections
    xrotorInputDict['a0deg'] = [0] * nAeroSections
    xrotorInputDict['dclda'] = [0] * nAeroSections
    xrotorInputDict['clmax'] = [0] * nAeroSections
    xrotorInputDict['clmin'] = [0] * nAeroSections
    xrotorInputDict['dcldastall'] = [0] * nAeroSections
    xrotorInputDict['dclstall'] = [0] * nAeroSections
    xrotorInputDict['mcrit'] = [0] * nAeroSections
    xrotorInputDict['cdmin'] = [0] * nAeroSections
    xrotorInputDict['clcdmin'] = [0] * nAeroSections
    xrotorInputDict['dcddcl2'] = [0] * nAeroSections

    for i in range(nAeroSections):
        comment_line = fid.readline().upper().split()
        check_comment(comment_line, 2)
        values = fid.readline().split()
        check_num_values(values, 1)
        xrotorInputDict['rRstations'][i] = float(values[0])

        comment_line = fid.readline().upper().split()
        check_comment(comment_line, 5)
        values = fid.readline().split()
        check_num_values(values, 4)
        xrotorInputDict['a0deg'][i] = float(values[0])
        xrotorInputDict['dclda'][i] = float(values[1])
        xrotorInputDict['clmax'][i] = float(values[2])
        xrotorInputDict['clmin'][i] = float(values[3])

        comment_line = fid.readline().upper().split()
        check_comment(comment_line, 5)
        values = fid.readline().split()
        check_num_values(values, 4)
        xrotorInputDict['dcldastall'][i] = float(values[0])
        xrotorInputDict['dclstall'][i] = float(values[1])
        xrotorInputDict['mcrit'][i] = float(values[3])

        comment_line = fid.readline().upper().split()
        check_comment(comment_line, 4)
        values = fid.readline().split()
        check_num_values(values, 3)
        xrotorInputDict['cdmin'][i] = float(values[0])
        xrotorInputDict['clcdmin'][i] = float(values[1])
        xrotorInputDict['dcddcl2'][i] = float(values[2])

        comment_line = fid.readline().upper().split()
        check_comment(comment_line, 3)
        values = fid.readline().split()

    # skip the duct information
    fid.readline()
    fid.readline()

    # Now we are done with the various aero sections and we start
    # looking at blade geometry definitions
    print('h0')
    comment_line = fid.readline().upper().split()
    check_comment(comment_line, 3)
    print(comment_line)
    print('h1')
    values = fid.readline().split()
    check_num_values(values, 2)
    print('h2')
    
    nGeomStations = int(values[0])
    xrotorInputDict['nGeomStations'] = nGeomStations
    xrotorInputDict['nBlades'] = int(values[1])
    xrotorInputDict['rRGeom'] = [0] * nGeomStations
    xrotorInputDict['cRGeom'] = [0] * nGeomStations
    xrotorInputDict['beta0Deg'] = [0] * nGeomStations

    comment_line = fid.readline().upper().split()
    check_comment(comment_line, 5)

    # iterate over all the geometry stations
    for i in range(nGeomStations):

        values = fid.readline().split()
        check_num_values(values, 4)
        xrotorInputDict['rRGeom'][i] = float(values[0]) 
        xrotorInputDict['cRGeom'][i] = float(values[1])
        xrotorInputDict['beta0Deg'][i] = float(values[2])

    # Set the twist at the root to be 90 so that it is continuous on
    # either side of the origin. I.e Across blades' root. Also set
    # the chord to be 0 at the root
    if xrotorInputDict['rRGeom'][0] != 0:
        xrotorInputDict['rRGeom'].insert(0, 0.0)
        xrotorInputDict['cRGeom'].insert(0, 0.0)
        xrotorInputDict['beta0Deg'].insert(0, 90.0)
        xrotorInputDict['nGeomStations'] += 1


    # AdvanceRatio = Vinf/Vtip => Vinf/OmegaR
    xrotorInputDict['omegaDim'] = \
        xrotorInputDict['vel'] / (xrotorInputDict['adv'] * xrotorInputDict['rad'])
    xrotorInputDict['RPM'] = xrotorInputDict['omegaDim'] * 30 / pi


    return xrotorInputDict

# to print in colors to the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def generateTwists(xrotorDict, gridUnit):
    # generate the twists vector required from the BET input
    twistVec = []
    for i in range(xrotorDict['nGeomStations']):
        # dimensional radius we are at in grid unit
        r = xrotorDict['rRGeom'][i] * xrotorDict['rad'] / gridUnit  
        twist = xrotorDict['beta0Deg'][i]
        twistVec.append({'radius': r, 'twist': twist})

    return twistVec

def generateChords(xrotorDict, gridUnit):
    # generate the dimensional chord vector required from the BET input
    chordVec = []
    for i in range(xrotorDict['nGeomStations']):
        r = xrotorDict['rRGeom'][i] * xrotorDict['rad'] / gridUnit  
        chord = xrotorDict['cRGeom'][i] * xrotorDict['rad'] / gridUnit  
        chordVec.append({'radius': r, 'chord': chord})

    return chordVec

def generateMachs(xrotorDict):
    # The Flow360 BET input file expects a set of Mach numbers to interpolate
    # between using the Mach number the blade sees.
    # To that end we will generate 4 different tables at 4 different Mach #s
    # equivalent to M^2=0, 1/3, 2/3, 0.9
    machVec = [0, sqrt(1 / 3), sqrt(2 / 3), sqrt(0.9)]
    return machVec

def generateAlphas(xrotorDict):
    # Generate the list of Alphas that the 2d section polar is for:
    # 1 degree steps from -180 to 180
    return list(arange(-180, 181, 1).astype(float))

def findClMinMaxAlphas(CLIFT, CLMIN, CLMAX):
    # find the index in the CLIFT array where we are just below the CLMin
    # value and jsut above hte CLmax value. Use the fact that CL is continually
    # increasing from -pi -> Pi radians. traverse the list until we hit CLMIN
    clMinIdx = 0
    clMaxIdx = len(CLIFT)
    for i in range(len(CLIFT)):
        if CLIFT[i] < CLMIN:
            clMinIdx = i
        if CLIFT[i] > CLMAX:
            clMaxIdx = i
            break
    return clMinIdx - 1, clMaxIdx + 1

def BlendFuncValue(blendWindow, alpha, alphaMinMax, plusorMinus):
    # this function is 1 at the alpha value and 0 at the
    # alpha+-blendwindow value
    
    if plusorMinus == 1:
        # we are on the CLMAX side:
        if alpha < alphaMinMax:
            return 1
        if alpha > alphaMinMax + blendWindow:
            return 0
        return cos((alpha - alphaMinMax) / blendWindow * pi / 2) ** 2
    if plusorMinus == -1:
        # we are on the CLMIN side:
        if alpha > alphaMinMax:
            return 1
        if alpha < alphaMinMax - blendWindow:
            return 0
        return cos((alpha - alphaMinMax) / blendWindow * pi / 2) ** 2

def blend2flatPlate(CLIFT, CDRAG, alphas, alphaMinIdx, alphaMaxIdx):
    # Blend the Clift and Cdrag values outside of the normal working
    # range of alphas to the flat plate CL and CD values.
    blendWindow = 0.5 
    alphaMin = alphas[alphaMinIdx] * pi / 180
    alphaMax = alphas[alphaMaxIdx] * pi / 180

    for i in range(alphaMinIdx):
        a = alphas[i] * pi / 180
         # -1 is b/c we are on the alphaCLmin side going up in CL
        blendVal = BlendFuncValue(blendWindow, a, alphaMin, -1) 
        CLIFT[i] = CLIFT[i] * blendVal + (1 - blendVal) * cos(a) * \
            2 * pi * sin(a) / sqrt(1 + (2 * pi * sin(a)) ** 2)
        CDRAG[i] = CDRAG[i] * blendVal + (1 - blendVal) * sin(a) * \
            (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6) + 0.05

    # from alphaMax to Pi in the CLIFT array
    for j in range(alphaMaxIdx, len(alphas)):
        a = alphas[j] * pi / 180  # alpha in radians
        blendVal = BlendFuncValue(blendWindow, a, alphaMax, 1)  
        CLIFT[j] = CLIFT[j] * blendVal + (1 - blendVal) * cos(a) * \
            2 * pi * sin(a) / sqrt(1 + (2 * pi * sin(a)) ** 2)
        CDRAG[j] = CDRAG[j] * blendVal + (1 - blendVal) * sin(a) * \
            (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6) + 0.05
    return CLIFT, CDRAG

def calcClCd(xrotorDict, alphas, machNum, nrRstation):
    # use the 2D polar parameters from the Xrotor input file to get the
    # Cl and Cd at the various Alphas and given MachNum

    # calculate compressibility factor taken from xaero.f in xrotor source code
    #  Factors for compressibility drag model, HHY 10/23/00
    #  Mcrit is set by user ( ie read in form Xrotor file )
    #  Effective Mcrit is Mcrit_eff = Mcrit - CLMFACTOR*(CL-CLDmin) - DMDD
    #  DMDD is the delta Mach to get CD=CDMDD (usually 0.0020)
    #  Compressible drag is CDC = CDMFACTOR*(Mach-Mcrit_eff)^MEXP
    # CDMstall is the drag at which compressible stall begins

    CDMFACTOR = 10.0
    CLMFACTOR = 0.25
    MEXP = 3.0
    CDMDD = 0.0020
    CDMSTALL = 0.1000

    # Prandtl-Glauert compressibility factor
    MSQ = machNum ** 2 

    if MSQ > 1.0:
        print('CLFUNC: Local Mach^2 number limited to 0.99, was ', MSQ)
        MSQ = 0.99

    PG = 1.0 / sqrt(1.0 - MSQ)
    MACH = machNum

    # Generate CL from dCL/dAlpha and Prandtl-Glauert scaling
    A_zero = xrotorDict['a0deg'][nrRstation] * pi / 180
    A0 = array([A_zero for i in range(len(alphas))])
    DCLDA = xrotorDict['dclda'][nrRstation]
    CLA = DCLDA * PG * ((array(alphas) * pi / 180) - A0)

    # Reduce CLmax to match the CL of onset of serious compressible drag
    CLMAX = xrotorDict['clmax'][nrRstation]
    CLMIN = xrotorDict['clmin'][nrRstation]
    CLDMIN = xrotorDict['clcdmin'][nrRstation]
    MCRIT = xrotorDict['mcrit'][nrRstation]

    DMSTALL = (CDMSTALL / CDMFACTOR) ** (1.0 / MEXP)
    CLMAXM = max(0.0, (MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMAX = min(CLMAX, CLMAXM)
    CLMINM = min(0.0, - (MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMIN = max(CLMIN, CLMINM)

    # CL limiter function (turns on after +-stall)
    DCL_STALL = xrotorDict['dclstall'][nrRstation]
    ECMAX = exp(clip((CLA - CLMAX) / DCL_STALL, None, 200))
    ECMIN = exp(clip((CLMIN - CLA) / DCL_STALL, None, 200))
    CLLIM = DCL_STALL * log((1.0 + ECMAX) / (1.0 + ECMIN))

    # Subtract off a (nearly unity) fraction of the limited CL function
    # This sets the dCL/dAlpha in the stalled regions to 1-FSTALL of that
    # in the linear lift range
    DCLDA_STALL = xrotorDict['dcldastall'][nrRstation]
    FSTALL = DCLDA_STALL / DCLDA
    CLIFT = CLA - (1.0 - FSTALL) * CLLIM

    # In the basic linear lift range drag is a quadratic function of lift
    # CD = CD0 (constant) + quadratic with CL)
    CDMIN = xrotorDict['cdmin'][nrRstation]
    DCDCL2 = xrotorDict['dcddcl2'][nrRstation]

    # Don't do any reynolds number corrections b/c we know it is minimal
    RCORR = 1
    CDRAG = (CDMIN + DCDCL2 * (CLIFT - CLDMIN) ** 2) * RCORR

    # Post-stall drag added
    FSTALL = DCLDA_STALL / DCLDA
    DCDX = (1.0 - FSTALL) * CLLIM / (PG * DCLDA)
    DCD = 2.0 * DCDX ** 2

    # Compressibility drag (accounts for drag rise above Mcrit with CL effects
    # CDC is a function of a scaling factor*(M-Mcrit(CL))**MEXP
    # DMDD is the Mach difference corresponding to CD rise of CDMDD at MCRIT
    DMDD = (CDMDD / CDMFACTOR) ** (1.0 / MEXP)
    CRITMACH = MCRIT - CLMFACTOR * abs(CLIFT - CLDMIN) - DMDD
    CDC = [0 for i in range(len(CRITMACH))]
    for critMachIdx in range(len(CRITMACH)):
        if (MACH < CRITMACH[critMachIdx]):
            continue
        else:
            CDC[critMachIdx] = CDMFACTOR * (MACH - CRITMACH[critMachIdx]) ** MEXP

    # you could use something like this to add increase drag by Prandtl-Glauert
    # (or any function you choose)
    FAC = 1.0
    # --- Total drag terms
    CDRAG = FAC * CDRAG + DCD + CDC

    # Now we modify the Clift and CDrag outside of the large alpha range to smooth out
    # the Cl and CD outside of the expected operating range

    # Find the Alpha for ClMax and CLMin
    alphaMinIdx, alphaMaxIdx = findClMinMaxAlphas(CLIFT, CLMIN, CLMAX)
    # Blend the CLIFt and CDRAG values from above with the flat plate formulation to
    # be used outside of the alphaCLmin to alphaCLMax window
    CLIFT, CDRAG = blend2flatPlate(CLIFT, CDRAG, alphas, alphaMinIdx, alphaMaxIdx)

    return list(CLIFT), list(CDRAG)

def getPolar(xrotorDict, alphas, machs, rRstation):
    # return the 2D Cl and CD polar expected by the Flow360 BET model.
    # b/c we have 4 Mach Values * 1 Reynolds value we need 4 different arrays per sectional polar as in:
    # since the order of brackets is Mach#, Rey#, Values then we need to return:
    # [[[array for MAch #1]],[[array for MAch #2]],[[array for MAch #3]],[[array for MAch #4]]]

    secpol = {}
    secpol['liftCoeffs'] = []
    secpol['dragCoeffs'] = []
    for machNum in machs:
        cl, cd = calcClCd(xrotorDict, alphas, machNum, rRstation)
        secpol['liftCoeffs'].append([cl])
        secpol['dragCoeffs'].append([cd])
    return secpol


def generateBETJSON(xrotorFileName, axisOfRotation, centerOfRotation,
                    rotationDirectionRule, **kwargs):

    diskThickness = kwargs['diskThickness']
    gridUnit = kwargs['gridUnit']
    chordRef = kwargs.pop('chordRef', 1.0)
    nLoadingNodes = kwargs.pop('nLoadingNodes', 20)
    tipGap = kwargs.pop('tipGap', 'inf')
    bladeLineChord = kwargs.pop('bladeLineChord', 0)
    initialBladeDirection = kwargs.pop('initialBladeDirection', [1, 0, 0])
    
#def generateBETJSON(xrotorFileName, axisOfRotation, centerOfRotation,
#                    rotationDirectionRule, diskThickness, chordRef,
#                    gridUnit, nLoadingNodes = 20, tipGap = 'inf',
#                   bladeLineChord = 0, initialBladeDirection = [1, 0, 0]):

    if rotationDirectionRule not in ['rightHand', 'leftHand']:
        print('Invalid rotationDirectionRule of {}. Exiting.'.format(rotationDirectionRule))

    if len(axisOfRotation) != 3:
        print('axisOfRotation must be a list of size 3. Exiting.')

    if len(centerOfRotation) != 3:
        print('centerOfRotation must be a list of size 3. Exiting')

    
    xrotorDict = readXROTORFile(xrotorFileName)

    diskJSON = {'axisOfrotation' : axisOfRotation,
                'centerOfRotation' : centerOfRotation,
                'rotationRule' : rotationDirectionRule}

    xrotorInflowMach = xrotorDict['vel'] / xrotorDict['vso']
    print('XROTOR inflow mach number: {}'.format(xrotorInflowMach))

    diskJSON['omega'] = xrotorDict['omegaDim'] * gridUnit / xrotorDict['vso'] # check this 
    diskJSON['numberOfBlades'] = xrotorDict['nBlades']
    diskJSON['radius'] = xrotorDict['rad'] / gridUnit
    diskJSON['omega'] = xrotorDict['omegaDim'] * gridUnit / xrotorDict['vso']
    diskJSON['twists'] = generateTwists(xrotorDict, gridUnit)
    diskJSON['chords'] = generateChords(xrotorDict, gridUnit)
    diskJSON['MachNumbers'] = generateMachs(xrotorDict)
    diskJSON['alphas'] = generateAlphas(xrotorDict)
    diskJSON['ReynoldsNumbers'] = [1.0]
    diskJSON['thickness'] = diskThickness
    diskJSON['chordRef'] = chordRef
    diskJSON['bladeLineChord'] = bladeLineChord
    diskJSON['nLoadingNodes'] = nLoadingNodes
    diskJSON['tipGap'] = tipGap
    diskJSON['sectionalRadiuses'] = [diskJSON['radius']*r for r in xrotorDict['rRstations']]
    diskJSON['initialBladeDirection'] = initialBladeDirection
    diskJSON['sectionalPolars'] = []
    print(diskJSON)
    for secId in range(0, xrotorDict['nAeroSections']):
        polar = getPolar(xrotorDict, diskJSON['alphas'], diskJSON['MachNumbers'], secId)
        diskJSON['sectionalPolars'].append(polar)

    return diskJSON
