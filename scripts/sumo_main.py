from sumolib import checkBinary
import numpy as np
import os, sys
from sumolib import net, xml
from random import randrange
import json
import math
import random
import utils
import traci
import argparse
import sumolib
import xml.etree.ElementTree as ET


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    # print(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no-gui", action='store_true',
                        help="run without the SUMO gui")

    return parser.parse_args()

def randomAssignmentOfPriority(npc_vehicleID,rl_vehicleID):
    for npc in npc_vehicleID:
        r = random.randint(0,10)
        if r > 7:
           traci.vehicle.setType(npc,"passenger-priority") 
        #    print("Vehicle ID:" + str(npc) + " assigned priority")
    for rl in rl_vehicleID:
        r = random.randint(0,10)
        if r > 7:
           traci.vehicle.setType(rl,"rl-default") 
        #    print("RLAgent ID:" + str(rl) + " assigned default")

def getState(net,agent_id):    
    agent_id = "RL_0"
    #Get the edgeID on which the RL agent is:
    edge_id = traci.vehicle.getRoadID(agent_id)
    print(edge_id)
    #Get the intersection the RL agent is going towards:
    # retrieve the successor edges of an edge
    nextEdges = net.getEdge(edge_id).getOutgoing()
    edge_list = [e.getID() for e in nextEdges] # list of all edges excluding internal
    priorityVehicleCount = 0
    nonPriorityVehicleCount = 0
    total_waiting_time = 0
    accumulated_time_loss = 0
    for edge_id in edge_list:   
        all_vehicle = traci.edge.getLastStepVehicleIDs(edge_id)
        for veh in all_vehicle:
            accumulated_time_loss+=traci.vehicle.getTimeLoss(veh)
            priority_type = traci.vehicle.getTypeID(veh)
            if priority_type=="passenger-priority" or priority_type=="rl-priority":
                priorityVehicleCount+=1
            else:
                nonPriorityVehicleCount+=1
        total_waiting_time+=traci.edge.getWaitingTime(edge_id)
        
        

    if traci.vehicle.getTypeID(agent_id)=="rl-priority":
        itsPriorityAccess = 1
    else:
        itsPriorityAccess = 0
    state = [itsPriorityAccess,priorityVehicleCount,nonPriorityVehicleCount,total_waiting_time,accumulated_time_loss]

    return state


def testCode(networkFileName):
    network = sumolib.net.readNet(networkFileName)
    numberOfRoutes = 28
    releventEdgeId = []
    allEdgeIds = traci.edge.getIDList()

    for i in range (numberOfRoutes): #number of routes to generate
        for edge in allEdgeIds:
            if edge.find("_") == -1:
                releventEdgeId.append(edge)
        rand_origin = random.choice(releventEdgeId) #choose a random origin
        releventEdgeId.remove(rand_origin)   #temporarily remove from the list 
        rand_dest_list = []
        for j in range (40): #number of intermediate destinations to generate
            dest_i = random.choice(releventEdgeId)
            rand_dest_list.append(dest_i) #choose a random destination
            releventEdgeId.remove(dest_i)
        path_list = []
        for dest_inter in rand_dest_list:
            path = traci.simulation.findRoute(str(rand_origin),str(dest_inter))
            rand_origin = dest_inter
            path_list.append(path)        
        #combine path
        combine_path = combinePath(path_list)


        print(combine_path)
        print("\n")
def computeEdges():
    allEdgeIds = traci.edge.getIDList()
    releventEdgeId = []
    for edge in allEdgeIds:
        if edge.find("_") == -1:
            releventEdgeId.append(edge)
    rand_origin = random.choice(releventEdgeId) #choose a random origin
    releventEdgeId.remove(rand_origin)   #temporarily remove from the list 
    rand_dest_list = []
    for j in range (40): #number of intermediate destinations to generate
        dest_i = random.choice(releventEdgeId)
        rand_dest_list.append(dest_i) #choose a random destination
        releventEdgeId.remove(dest_i)
    path_list = []
    for dest_inter in rand_dest_list:
        path = traci.simulation.findRoute(str(rand_origin),str(dest_inter))
        rand_origin = dest_inter
        path_list.append(path)        
    #combine path
    combine_path = combinePath(path_list)
    return combine_path

def createNPCRouteFiles(networkFileName):
    routeFileName = "sumo_configs/heuristic.rou.generated.xml"   

    root = ET.Element('vehicles')
    numberOfRoutes = 50
    releventEdgeId = []
    allEdgeIds = traci.edge.getIDList()

    for i in range (numberOfRoutes): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        npc_id = f"heuristic_" + str(i)
        assignPriority = random.uniform(0, 1)
        if assignPriority > 0.5:
            priorityType = "heuristic-priority"
        else:
            priorityType = "heuristic-default"

        
        edges = computeEdges()
        child.set('id',str(npc_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))


    # print(tostring(top))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(routeFileName)

def readRandomTripGeneratedRouteFileAndCreateRoutesForMultipleVehicleType(networkFileName):
    routeFileName = "sumo_configs/long_routes.rou.xml"
    newRouteFileName = "sumo_configs/all_routes.rou.xml"
    routesList = []
    tree = ET.parse(routeFileName)
    root = tree.getroot()
    
    for route in root.iter('route'):
        routesList.append(route.attrib['edges'])

    
    rl_agent_numer = 28
    root = ET.Element('vehicles')
    for i in range (rl_agent_numer): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        rl_id = f"RL_" + str(i)       
        priorityType = "rl-priority"  
        list_length =  len(routesList)
        random_index = randrange(list_length)
        edges = routesList[random_index]
        child.set('id',str(rl_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))
        routesList.remove(edges)
    
    cav_agent_numer = 40
    for i in range (cav_agent_numer): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        cav_id = f"cav_" + str(i)       
        priorityType = "cav-priority"  
        list_length =  len(routesList)
        random_index = randrange(list_length)
        edges = routesList[random_index]
        child.set('id',str(cav_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))
        routesList.remove(edges)

    heuristic_agent_numer = 50
    for i in range (heuristic_agent_numer): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        heuristic_id = f"heuristic_" + str(i) 
        assignPriority = random.uniform(0, 1)
        if assignPriority > 0.5:
            priorityType = "heuristic-priority"
        else:
            priorityType = "heuristic-default"  
        list_length =  len(routesList)
        random_index = randrange(list_length)
        edges = routesList[random_index]
        child.set('id',str(heuristic_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))
        routesList.remove(edges)
    
    npc_agent_numer = 50
    for i in range (npc_agent_numer): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        npc_id = f"npc_" + str(i) 
        assignPriority = random.uniform(0, 1)
        if assignPriority > 0.5:
            priorityType = "passenger-priority"
        else:
            priorityType = "passenger-default"  
        list_length =  len(routesList)
        random_index = randrange(list_length)
        edges = routesList[random_index]
        child.set('id',str(npc_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))
        routesList.remove(edges)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(newRouteFileName)


def createCAVRouteFiles(networkFileName):
    routeFileName = "sumo_configs/cav.rou.generated.xml"   

    root = ET.Element('vehicles')
    numberOfRoutes = 40
    releventEdgeId = []
    allEdgeIds = traci.edge.getIDList()

    for i in range (numberOfRoutes): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        npc_id = f"cav_" + str(i)       
        priorityType = "cav-priority"        
        edges = computeEdges()
        child.set('id',str(npc_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))


    # print(tostring(top))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(routeFileName)

def combinePath(path_list):
    combinePath = []
    for path in path_list:
        edge_list = []
        edge_list = path.edges
        combinePath += edge_list
        combinePath.pop()
    cleanString = ""
    for p in combinePath:
        p.replace("'","")
        cleanString +=str(p) + " "
    return cleanString
        
       


def init_simulator(seed,networkFileName,withGUI):
    sumoCMD = ["--seed", str(seed), "-W","--default.carfollowmodel", "IDM","--no-step-log","--statistic-output","output.xml"]
 
  
    if withGUI:
        sumoBinary = checkBinary('sumo-gui')
        # sumoCMD += ["--start", "--quit-on-end"]
        sumoCMD += ["--start","--quit-on-end"]
    else:
        sumoBinary = checkBinary('sumo')

    # print(sumoBinary)
    sumoConfig = "sumo_configs/sim.sumocfg"
    sumoCMD = ["-c", sumoConfig] + sumoCMD


    random.seed(seed)
    traci.start([sumoBinary] + sumoCMD)
    episodeLength = 3600
    stepCounter = 0
    warmUpPeriod = 300


    
    all_traffic = ['pedestrian','private', 'emergency', 'passenger','authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']

    # Access all npc vehicles and randomly change the priority access
    
    releventEdgeId = []
    allEdgeIds = traci.edge.getIDList()
    for edge in allEdgeIds:
        if edge.find("_") == -1:
            releventEdgeId.append(edge)

    # readRandomTripGeneratedRouteFileAndCreateRoutesForMultipleVehicleType(networkFileName)
    # testCode(networkFileName)
    # createNPCRouteFiles(networkFileName)
    # createCAVRouteFiles(networkFileName)
    # print("Rohit")
    # print(releventEdgeId)
    while stepCounter < episodeLength:
        traci.simulationStep(stepCounter)
        #### WARM UP PERIOD CODE HERE ####
        # if stepCounter > 10:
           
            # allVehicleList = traci.vehicle.getIDList()
            # npc_vehicleID,rl_vehicleID = utils.getSplitVehiclesList(allVehicleList)
            # print(rl_vehicleID)
            # for rl_veh in rl_vehicleID:
            #     if traci.vehicle.getRouteIndex(str(rl_veh)) == (len(traci.vehicle.getRoute(str(rl_veh))) - 1): #Check to see if the car is at the end of its route
            #         new_destiny = random.choice(releventEdgeId)
            #         # print(str(i)+str(new_destiny))
            #         traci.vehicle.changeTarget(str(rl_veh),str(new_destiny)) #Assign random destination

        # if stepCounter < 300:
            # print("Inside Warm-up Period")
        #### ACTION STEP CODE HERE ####
        if stepCounter%300 == 0 and stepCounter>=warmUpPeriod:
            print("Inside Action Step")
            print(len(traci.vehicle.getIDList()))
            # npc_vehicleID,rl_vehicleID = utils.getSplitVehiclesList(allVehicleList)
            # print("Total npc: " + str(len(npc_vehicleID)) + "Total RL agent: " + str(len(rl_vehicleID)))
            # randomAssignmentOfPriority(npc_vehicleID,rl_vehicleID)

            # print(getState(net))
            
        stepCounter +=1
    traci.close()

   
if __name__ == "__main__":
    args = parse_args()

    """
    Configure various parameters of SUMO
    """
    withGUI = not args.no_gui

    if not withGUI:
        try:
            import libsumo as traci
        except:
            pass
    
    seed = 42
    networkFileName = "sumo_configs/Grid1.net.xml"
    init_simulator(seed,networkFileName,True)

   
