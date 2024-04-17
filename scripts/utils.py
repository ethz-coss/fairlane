import xml.etree.ElementTree as ET
import os
import sys
import numpy as np

from gym.spaces import Box, Discrete

def get_space_dims(space):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        return space.shape[0]
    else:
        raise ValueError("space must be Discrete, Box")
    
def editLaneVClassAllowedPermission(toNew, networkFileName="sumo_configs/Grid1.net.xml"):
    print(toNew)
    tree = ET.parse(networkFileName)
    root = tree.getroot()

    for lanes in root.iter('lane'):
        if lanes.attrib['index'] == "2":
            lanes.set("allow",toNew)

    
    #  write xml 
    file_handle = open(networkFileName,"wb")
    tree.write(file_handle)
    file_handle.close()


