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

def getSplitVehiclesList(allvehicles):
    rl_vehicleID = []
    cav_vehicleID = []
    heuristic_vehicleID = []
    npc_vehicleID = []
    for veh in allvehicles:
        x = veh.split("_",1)
        if x[0] =="RL":
            rl_vehicleID.append(veh)
        elif x[0] == "cav":
            cav_vehicleID.append(veh)
        elif x[0] == "heuristic":
            heuristic_vehicleID.append(veh)
        elif x[0] == "npc":
            npc_vehicleID.append(veh)
    return npc_vehicleID,rl_vehicleID,heuristic_vehicleID,cav_vehicleID
