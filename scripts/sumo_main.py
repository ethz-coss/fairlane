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
from copy import deepcopy
import subprocess

# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
#     # print(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")



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


def computeEdges(index, traci,allEdgeIds):
    traci = traci
    allEdgeIds = allEdgeIds
    releventEdgeId = []
    for edge in allEdgeIds:
        if edge.find("_") == -1:
            releventEdgeId.append(edge)

    veh_id = f"cav_" + str(index)
    traci.vehicle.add(veh_id, '', typeID='cav-priority', depart=0)
    init_edge = np.random.choice(releventEdgeId)
    pos = 0.0
    ## HACK
    route = [init_edge]
    # traci.vehicle.setVia(veh_id, route)
    # traci.vehicle.rerouteTraveltime(veh_id)
    traci.vehicle.setRoute(veh_id,route)
    traci.vehicle.moveTo(veh_id,f"{init_edge}_0",pos)
    r = traci.vehicle.getRoute(veh_id)
    traci.simulationStep()
    # size = np.random.randint(50, 80)
    # init_edge, = traci.vehicle.getRoute(veh_id) # a random edge was assigned
   
    route = np.random.choice(allEdgeIds, size=80, replace=True).tolist()
    route = [init_edge] + route
    traci.vehicle.setVia(veh_id, route)
    traci.vehicle.rerouteTraveltime(veh_id)
    combine_path = traci.vehicle.getRoute(veh_id)
    combine_path = list(combine_path)
    
    return combine_path

def createCAVRouteFiles(traci,networkFileName):
    traci = traci
    routeFileName = "sumo_configs/Test/cav.rou.generated.xml"   
    network = sumolib.net.readNet(networkFileName)
    allEdgeIds = [Edge.getID() for Edge in network.getEdges(withInternal=False)]
    root = ET.Element('vehicles')
    numberOfRoutes = 350

    for i in range (numberOfRoutes): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        cav_id = f"cav_" + str(i)
        priorityType = "cav-priority"        
        edges = computeEdges(i,traci,allEdgeIds)
        child.set('id',str(cav_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',' '.join(edges).replace("'",""))


    # print(tostring(top))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(routeFileName)

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
    routeFileName = "./sumo_configs/Test/Barcelona/Barcelona.rou.xml"
    newRouteFileName = "./sumo_configs/Test/Barcelona/Barcelona_raw.rou.xml"
    routesList = []
    tree = ET.parse(routeFileName)
    root = tree.getroot()
    
    for route in root.iter('route'):
        routesList.append(route.attrib['edges'])

    
    rl_agent_numer = 350
    root = ET.Element('vehicles')
    for i in range (rl_agent_numer): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        rl_id = f"RL_" + str(i)       
        priorityType = "rl-default"  
        list_length =  len(routesList)
        random_index = randrange(list_length)
        edges = routesList[random_index]
        child.set('id',str(rl_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))
        routesList.remove(edges)
    
    cav_agent_numer = 350
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

    # heuristic_agent_numer = 50
    # for i in range (heuristic_agent_numer): #number of routes to generate
    #     child = ET.SubElement(root, 'vehicle')
    #     #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
    #     heuristic_id = f"heuristic_" + str(i) 
    #     assignPriority = random.uniform(0, 1)
    #     if assignPriority > 0.5:
    #         priorityType = "heuristic-priority"
    #     else:
    #         priorityType = "heuristic-default"  
    #     list_length =  len(routesList)
    #     random_index = randrange(list_length)
    #     edges = routesList[random_index]
    #     child.set('id',str(heuristic_id))
    #     child.set('type',str(priorityType))
    #     child.set('depart',"0.0")
    #     route = ET.SubElement(child, 'route')
    #     route.set('edges',str(edges))
    #     routesList.remove(edges)
    
    npc_agent_numer = 350
    for i in range (npc_agent_numer): #number of routes to generate
        child = ET.SubElement(root, 'vehicle')
        #id="RL_0" type="rl-priority" depart="0.0"> <route edges="E1E0 E0D0 D0C0
        npc_id = f"npc_" + str(i)       
        priorityType = "passenger-default"  
        list_length =  len(routesList)
        random_index = randrange(list_length)
        edges = routesList[0]
        child.set('id',str(npc_id))
        child.set('type',str(priorityType))
        child.set('depart',"0.0")
        route = ET.SubElement(child, 'route')
        route.set('edges',str(edges))
        routesList.pop(0)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(newRouteFileName)

def readRouteFileAndCreateRoutesForMultipleVehicleType_OLD():
    routeFileName = "./sumo_configs/Test/Barcelona/Barcelona_raw.rou.xml"
    newRouteFileName = "./sumo_configs/Test/Barcelona/Barcelona.rou.xml"
    routesList = []
    tree = ET.parse(routeFileName)
    root = tree.getroot()
    
    for route in root.iter('route'):
        routesList.append(route.attrib['edges'])
    rl_counter = 0
    cav_counter = 0
    npc_counter = 0
    root_new = ET.Element('vehicles')
    for veh in root.iter('vehicle'):
    #randomly decide to assign vehicle type between CAV,RL,NPC
        n=random.randint(0,2)
        depart = veh.attrib['depart']
        depart = str(float(depart) - 26100)
        departPos = veh.attrib['departPos']
        departSpeed = veh.attrib['departSpeed']
        fromTaz = veh.attrib['fromTaz']
        toTaz = veh.attrib['toTaz']
        edges = routesList[0]
        route.set('edges',str(edges))
        routesList.pop(0)
        
        if n==0: #RL
            child = ET.SubElement(root_new, 'vehicle')
            rl_id = f"RL_" + str(rl_counter)
            rl_counter+=1
            priorityType = "rl-default" 
            child.set('id',str(rl_id))
            child.set('type',str(priorityType))            
           
        elif n==1: #CAV
            child = ET.SubElement(root_new, 'vehicle')
            cav_id = f"cav_" + str(cav_counter)
            cav_counter+=1
            priorityType = "cav-priority"
            child.set('id',str(cav_id))
            child.set('type',str(priorityType)) 

        elif n==2: #NPC
            child = ET.SubElement(root_new, 'vehicle')
            npc_id = f"npc_" + str(npc_counter)
            npc_counter+=1
            priorityType = "passenger-default"
            child.set('id',str(npc_id))
            child.set('type',str(priorityType))   

        child.set('depart',depart)
        child.set('departPos',departPos)
        child.set('departSpeed',departSpeed)
        child.set('fromTaz',fromTaz)
        child.set('toTaz',toTaz)
        route = ET.SubElement(child, 'route')
        route.set('edges',edges)

    tree = ET.ElementTree(root_new)
    ET.indent(tree, space="\t", level=0)
    tree.write(newRouteFileName)

def scaleRouteFile():
    basefile = "./sumo_configs/Test/Barcelona/route_basefile.rou.xml"
    routeFileName = "./sumo_configs/Test/Barcelona/Barcelona_raw.rou.xml"
    newRouteFileName = "./sumo_configs/Test/Barcelona/Barcelona.rou.xml"
    routesList = []
    root = ET.parse(routeFileName).getroot()
    
    root_new = ET.parse(basefile).getroot()

    veh_types = {'cav': 'cav-priority',
                 'npc': 'passenger-default',
                 'RL': 'rl-default'}
    for i, veh in enumerate(root.iter('vehicle')):
        for veh_type, priorityType in veh_types.items():
            new_veh = deepcopy(veh)
            depart = new_veh.attrib['depart']
            depart = str(float(depart) - 26100)
            new_veh.attrib['depart'] = depart
            new_veh.set('id', f'{veh_type}_{i}')
            new_veh.set('type', priorityType)
            new_veh.attrib.pop('departSpeed', None)
            new_veh.attrib.pop('departPos', None)
            root_new.append(new_veh)

    tree = ET.ElementTree(root_new)
    ET.indent(tree, space="\t", level=0)
    tree.write(newRouteFileName)

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
    
    withGUI = False
    if withGUI:
        import traci            
    else:
        try:
            import libsumo as traci
        except:
            import traci
   
  
    sumoConfig = "sumo_configs/Test/MSN_Grid.sumocfg"

    sumoCMD = ["-c", sumoConfig, "--time-to-teleport", str(-1),"--scale",str(1),
				"-W","--collision.action","none"]
    if withGUI:
        sumoBinary = checkBinary('sumo-gui')
        sumoCMD += ["--start"]
    else:	
        sumoBinary = checkBinary('sumo')

    
    traci.start([sumoBinary] + sumoCMD)
    return traci
    


def basic_routing_adapt(INPUT_ROUTES_PATH, NEW_NET_FILE, OUTPUT_ROUTE_FILE, write_trips=False):
    
    # OLD_ROUTES_PATH = r"{}/{}".format(OLD_SCENARIO_FOLDER, OLD_ROUTES_FILE)
    # NEW_NET_PATH = r"{}/{}".format(NEW_SCENARIO_FOLDER, NEW_NET_FILE)
    OLD_ROUTES_PATH = r"{}".format(INPUT_ROUTES_PATH)
    NEW_NET_PATH = r"{}".format(NEW_NET_FILE)
    
    duarouter = sumolib.checkBinary('duarouter')
    base_command = f"{duarouter} --net-file {NEW_NET_PATH} --route-files {INPUT_ROUTES_PATH} --routing-algorithm astar --routing-threads {8} -W --ignore-errors --repair --repair.from --repair.to --weights.priority-factor 10"
    if write_trips==False:
        # NEW_ROUTES_PATH = r"{}/{}".format(NEW_SCENARIO_FOLDER, OLD_ROUTES_FILE)
        NEW_ROUTES_PATH = r"{}".format(OUTPUT_ROUTE_FILE)
        subprocess.run(f"{duarouter} --net-file {NEW_NET_PATH} --route-files {INPUT_ROUTES_PATH} --output-file {NEW_ROUTES_PATH} --routing-algorithm astar --routing-threads {8} -W --ignore-errors --repair --repair.from --repair.to --weights.priority-factor 10".format(NEW_NET_PATH, INPUT_ROUTES_PATH, NEW_ROUTES_PATH, 8))
    else:
        # NEW_ROUTES_PATH = r"{}/{}.trips.rou.xml".format(NEW_SCENARIO_FOLDER, OLD_ROUTES_FILE.split(".")[0])
        NEW_ROUTES_PATH = r"{}.trips.rou.xml".format(OUTPUT_ROUTE_FILE.split(".")[0])
        base_command += " --write-trips"
    base_command += f" --output-file {NEW_ROUTES_PATH}"
    subprocess.Popen(base_command, shell=True)
    
    return print("Routing FIXED/ADAPTED from \n{} \ndone in \n{} \nfor \n{}".format(OLD_ROUTES_PATH, NEW_ROUTES_PATH, NEW_NET_PATH))

    

if __name__ == "__main__":   
    
    seed = 42
    # networkFileName = "sumo_configs/Test/MSN_Grid_rebuildTrafficLight.net.xml"
    # traci = init_simulator(seed,networkFileName,True)
    
    # createCAVRouteFiles(traci,networkFileName)
    scaleRouteFile()
    print(os.getcwd())
    oldRouteFileName = "./sumo_configs/Test/Barcelona/Barcelona.rou.xml"
    newRouteFileName = "./sumo_configs/Test/Barcelona/Barcelona_scaled.rou.xml"
    netFileName = "./sumo_configs/Test/Barcelona/Barcelona.net.xml"
    basic_routing_adapt(oldRouteFileName, netFileName, newRouteFileName) # somehow does not work in python
    # times = 3 #meaning 3 RL vehicle instead of 1 with different departTime
    # scaleRouteFile(times)

   
