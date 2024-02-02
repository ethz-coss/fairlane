import numpy as np
import os, sys
from addLoopDetectors import loopDetector
from sumolib import net

loopDetectorFileName = "C:/D/SUMO/PriorityLane/sumo_configs/LTN_loopDetectors.add.xml"
networkFileName = 'C:/D/SUMO/PriorityLane/sumo_configs/LargeTestNetwork.net.xml'


step = 0

network = net.readNet(networkFileName)

edge_list = [e.getID() for e in network.getEdges(withInternal=False)] # list of all edges excluding internal links

# Write additional file with loop detectors if the file does not exist

###uncomment below function everytime you need to generate new Loop  detector additional file
loopDetector(network, edge_list, loopDetectorFileName)
###uncomment below function everytime you need to generate new Loop  detector additional file

