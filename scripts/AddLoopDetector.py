import numpy as np
import os, sys
from addLoopDetectors import loopDetector
from sumolib import net

loopDetectorFileName = "D:/prioritylane/sumo_configs/GridRectange.add.xml"
networkFileName = 'D:/prioritylane/sumo_configs/GridRectangle.net.xml'


step = 0

network = net.readNet(networkFileName)

edge_list = [e.getID() for e in network.getEdges(withInternal=False)] # list of all edges excluding internal links

# Write additional file with loop detectors if the file does not exist

###uncomment below function everytime you need to generate new Loop  detector additional file
loopDetector(network, edge_list, loopDetectorFileName)
###uncomment below function everytime you need to generate new Loop  detector additional file

