import json
import sys
import os
import unittest
testDir = os.path.dirname(os.path.abspath(__file__))

sys.path.append('{}/../preprocessing/xrotor'.format(testDir))
from xrotorBETInterface import generateBETJSON

diskThickness = 0.2
chordRef = 0.2

betJSON = generateBETJSON('{0}/ref/xv15_airplane_pitch26.prop'.format(testDir),
                          [0, 0, 1], [0, 0, 0],
                          'rightHand', diskThickness=diskThickness,
                          chordRef=chordRef, gridUnit=1.0, nLoadingNodes=20, tipGap='inf')


refJSON = json.load(open('{0}/ref/{1}'.format(testDir, 'xv15_bet_ref.json')))
assert(refJSON == betJSON)

#json.dump(betJSON, open('xv15_bet_ref.json', 'w'), indent=4)

