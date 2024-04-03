from gym import Env
from gym import spaces
from gym.utils import seeding
from gym import spaces
import numpy as np
import math
from sumolib import checkBinary
import os, sys
import random
import traci
from scripts import utils
import xml.etree.ElementTree as ET
import math
from itertools import combinations
import sumolib
from sumolib import net
from lxml import etree as ET
from utils.common import convertToFlows

def generate_routefile(base_routefile, out_route_file, cav_rate, hdv_rate, n_agents, baseline):
	cav_period, npc_period, _n_agents = convertToFlows(cav_rate,hdv_rate)
	print(f'WARNING: n_agents not the same as output of convertToFlows: {n_agents} vs {_n_agents}',
	      'Following convertToFlows')
	data = ET.Element('routes')
	base_routes = ET.parse(base_routefile)
	vehicles = base_routes.findall('vehicle')
	vtypes = base_routes.findall('vType')
	flows = base_routes.findall('flow')
	for vtype in vtypes:
		if vtype.attrib['id']=='rl-priority':
			if baseline=='baseline1':
				vtype.attrib['maxSpeed'] = '13.89'
		data.append(vtype)
	for flow in flows:
		if flow.attrib['type']=='passenger-default':
			flow.attrib['period'] = f'exp({npc_period:.4f})'
		if flow.attrib['type']=='cav-priority':
			flow.attrib['period'] = f'exp({cav_period:.4f})'
		data.append(flow)
	for i, vehicle in enumerate(vehicles):
		if i==_n_agents:
			break
		data.append(vehicle)
	with open(out_route_file, "wb") as f:
		f.write(ET.tostring(data, pretty_print=True))

class Agent:
    def __init__(self, env, n_agent, edge_agent=None):
        """Dummy agent object"""
        self.edge_agent = edge_agent
        self.env = env

        self.id = n_agent
        self.name = f'RL_{self.id}'

class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	
	def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',testStatAccumulation=10,
				 testFlag='False',simulation_end=36000, num_agents=50, action_step=30,
				 episode_duration=None, cav_rate=10, hdv_rate=50, scenario_flag='model'):
		self.pid = os.getpid()
		self.sumoCMD = []
		self._simulation_end = simulation_end
		self._mode = mode
		self.SotaFlag = scenario_flag=='sota'
		self._testStatAccumulation = testStatAccumulation
		if testFlag == False:
			# self._networkFileName = "sumo_configs/LargeTestNetwork.net.xml"
			# self._routeFileName = "sumo_configs/LargeTestNetwork.rou.xml"   
			self._networkFileName = "sumo_configs/Train/GridNoInternalLink.net.xml"
			self._routeFileName = "sumo_configs/Train/routes.rou.xml"
			self._warmup_steps = 100
			self.sumoConfig = "sumo_configs/Train/sim.sumocfg"   

			# self._networkFileName = "sumo_configs/GridRectangle.net.xml"
			# self._routeFileName = "sumo_configs/GridRectangle.rou.xml"
			# self._warmup_steps = 900

		else:
			# self._networkFileName = "sumo_configs/Grid1.net.xml"
			# self._routeFileName = "sumo_configs/routes.rou.xml"  

			self._networkFileName = "sumo_configs/Test/MSN_Grid_rebuildTrafficLight.net.xml"
			self._baseRouteFileName = "sumo_configs/Test/MSN_Grid_base.rou.xml"
			self._routeFileName = f"sumo_configs/Test/rou_{cav_rate}_{hdv_rate}_{scenario_flag}.rou.xml"
			generate_routefile(self._baseRouteFileName, self._routeFileName, cav_rate, hdv_rate, num_agents, scenario_flag)
			self._warmup_steps = 300
			self.sumoConfig = "sumo_configs/Test/MSN_Grid.sumocfg"

			# self._networkFileName = "sumo_configs/GridNoInternalLink.net.xml"
			# self._routeFileName = "sumo_configs/routes.rou.xml"
			# self._warmup_steps = 100

			# self._networkFileName = "sumo_configs/GridRectangle.net.xml"
			# self._routeFileName = "sumo_configs/GridRectangle.rou.xml"
			# self._warmup_steps = 300
			

		# self.network = net.readNet(self._networkFileName)

		# self.edge_list = [e.getID() for e in self.network.getEdges(withInternal=False)]
		self._episodeStep = 0
		self._isTestFlag = testFlag
		self._rl_counter = 0
		self._cavFlowCounter = 0
		self._collisionCount = 0
		self._collisionVehicleID = []
		# self._seed(40)
		# np.random.seed(42)
		self._sumo_seed = 42
		self._reward_type = "Global" 
		# self._reward_type = "Local" 
		# self._reward_type = "Individual"
		self.withGUI = mode
		self.action_steps = action_step
		self.episode_duration = episode_duration
		
		self._sumo_step = 0		
		self.shared_reward = True
		self._fatalErroFlag = False
		self._alreadyAddedFlag = False
		self._scenario = "Train"
		self._npc_vehicleID = 0
		self._rl_vehicleID = 0
		self._heuristic_vehicleID=0
		self._cav_vehicleID=0
		self.original_rl_vehicleID = []
		self._routeDict = {}
		self._timeLossOriginalDict = {}
		self._statePerimeter = 65
		self._stateVehicleCount = 10
		self._n_features = 4
		self._net = sumolib.net.readNet(self._networkFileName,withInternal=True)
		self._allEdgeIds = [Edge.getID() for Edge in self._net.getEdges(withInternal=False)]
		# set required vectorized gym env property
		self.n = num_agents #read it from the route file
		self.lastActionDict = {}
		self.lastTimeLossRLAgents = {}
		self._trafficPhaseRLagent = {}
		self._lastOverAllTimeLoss = {}
		self._lastEdge = {}
		self._lastLane = {}
		self._nextLane = {}
		self._currentOverAllTimeLoss = {}
		self._lastOverAllWaitingTime = {}
		self._throughputAfter = {}
		self._throughputBefore = {}
		self._lastCAVWaitingTimeForSpecificRLAgent = {}
		self._currentCAVWaitingTimeForSpecificRLAgent = {}
		self._currentRLWaitingTimeForSpecificRLAgent = {}
		self._lastRLWaitingTimeForSpecificRLAgent = {}
		self._numberOfCAVWithinClearingDistanceOnPLAfter = {}
		self._currentOverAllWaitingTime = {}
		self._listOfVehicleIdsInConcern = {}
		self._numberOfCAVWithinClearingDistance = {}
		self._numberOfCAVWithinClearingDistanceBefore = {}
		self._numberOfCAVWithinClearingDistanceAfter = {}
		self._numberOfCAVApproachingIntersection = {}
		self._beforePriorityForRLAgent = {}
		self._afterPriorityForRLAgent = {}
		self._listOfLocalRLAgents = {}
		self._BeforeSpeed = {}
		self._AfterSpeed = {}
		self._BeforeCAVSpeed = {}
		self._AfterCAVSpeed = {}
  
		self._releventEdgeId = []
		self._rlLaneID = {}
		self._allVehLaneIDBefore = {}
		self._allVehLaneIDAfter = {}
		self._timeLossThreshold = 60
		self._lane_clearing_distance_threshold = 50
		self._lane_clearing_distance_threshold_RL = 5
		self._lane_clearing_distance_threshold_state = 50
		self._laneChangeAttemptDuration = 2 #seconds
		self._weightCAVPriority = 1 #3
		self._weightRLWeightingTime = 1
		self._weightCAVWeightingTime = 1 #2

		#test stats
		self._currentTimeLoss_rl = 0
		self._currentTimeLoss_npc = 0
		self._currentTimeLoss_cav = 0
		self._currentTimeLoss_Heuristic = 0
		self._avg_speed_rl = 0
		self._avg_speed_heuristic = 0
		self._avg_speed_npc = 0
		self._avg_speed_cav = 0		
		self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0
		self._average_edge_occupancy = 0
		self._average_priorityLane_occupancy = 0
		self._average_throughput = 0
		self._average_PMx_emission = 0
		self._average_LaneChange_number = 0
		self._average_LaneChange_number_all = 0
		self._collisionCounter = 0

		for edge in self._allEdgeIds:
			if edge.find(":") == -1:
				self._releventEdgeId.append(edge)

		
		#############------------------------------------------------------------------#######################

		# for every trip there is a hypothetical duration tMIN that could be achieved if the vehicle was driving with its maximum
		# allowed speed (including speedFactor) and there were no other vehicles nor traffic rules.
		# timeLoss = tripDuration - tMIN
		# also, waiting time is always included in timeLoss, therefore
		# timeLoss >= waitingTime

		#############------------------------------------------------------------------#######################



		# priority_actions = ['0','1','2']
		# configure spaces
		# self._num_observation = [len(self.getState(f'RL_{i}')) for i in range(self.n)]
		self._num_observation = 7
		# self._num_observation = self._n_features*self._stateVehicleCount + 2
		self._num_actions = 2
		# self._num_actions = [len(priority_actions), len(priority_actions)]
		# self._num_observation = [len(Agent(self, i, self.edge_agents[0]).getState()) for i in range(self._num_lane_agents)]*len(self.edge_agents)
		# self.action_space = spaces.MultiDiscrete([self._num_actions]*self.n)
		# self.observation_space = spaces.Box(low=0, high=1, shape=(self.n, self._num_observation,))
		self.action_space = spaces.Tuple([spaces.Discrete(self._num_actions) for i in range(self.n)])
		self.observation_space = spaces.Tuple([spaces.Box(low=0, high=1, shape=(self._num_observation,)) for i in range(self.n)])


		# for i in range(self.n):
		# 	self.action_space.append(spaces.Discrete(self._num_actions)) #action space			
		# 	self.observation_space.append(spaces.Box(low=0, high=1, shape=(self._num_observation,)))# observation space
			
		self.agents = self.createNAgents()
		self.controlled_vehicles = self.agents

		self.traci = self.initSimulator(self.withGUI, self.pid)

		# parse the net
		self.resetAllVariables()

	def createNAgents(self):
		agents = [Agent(self, i) for i in range(self.n)]
		return agents

	# def getState(self,agent_id):
	# 	"""
	# 	Retrieve the state of the network from sumo. 
	# 	"""
	# 	state = []
	# 	#detect all vehicle that is within the threshold distance for states. 
		
 		
	# 	#Get the edgeID on which the RL agent is:
	# 	edge_id = self.traci.vehicle.getRoadID(agent_id)
	# 	lane_id = self.traci.vehicle.getLaneID(agent_id)	
	# 	agent_speed = self.traci.vehicle.getSpeed(agent_id)
  
		
	# 	features = []
  	
	# 	# print(edge_id,agent_id)
	# 	nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
	# 	# now found the edges that is incoming to this junction
	# 	incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()
	# 	edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
	# 	occupancy = 0
	# 	n_external_edges_counter = 0
	# 	for e_id in edge_list_incoming: 
	# 		#only add non-internal road occupancy
	# 		if e_id.find(":") == -1:
	# 			occupancy+= self.traci.edge.getLastStepOccupancy(e_id)
	# 			n_external_edges_counter+=1
		
	# 	features.append(occupancy/n_external_edges_counter)

	# 	#count all  CAV vehicle behind this RL agent and within clearingThresholdDistance
	# 	agent_lane_pos = self.traci.vehicle.getLanePosition(agent_id)
	# 	agent_pos = self.traci.vehicle.getPosition(agent_id)
	# 	all_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
	# 	#loop through all vehicle and only store CAV and RL agents
	# 	cav_rl_agents = []
	# 	for v in all_vehicle:
	# 		priority_type = self.traci.vehicle.getTypeID(v)
	# 		if priority_type=="cav-priority" or priority_type=="rl-priority" or priority_type=="rl-default":
	# 			cav_rl_agents.append(v)
	# 	edge_length = self.traci.lane.getLength(lane_id)
	# 	agent_pos_fromIntersection = edge_length - agent_lane_pos
		
	# 	features.append(agent_speed/25)
	# 	# features.append(occupancyNextLane)
	# 	# features.append(occupancyCurrentLane)
	# 	# features.append(occupancyCurrentPriorityLane)
	# 	# features.append(occupancyNextPriorityLane)
  
	# 	cav_rl_agents.remove(agent_id)
	# 	if agent_pos_fromIntersection > self._statePerimeter:
	# 		#then only consider vehicle of single edge on which the agent is plying
	# 		#randomly select 10 vehicles from this list
	# 		# all_vehicle = list(dict.fromkeys(all_vehicle))
	# 		count = min(len(cav_rl_agents),self._stateVehicleCount)
	# 		random_list_vehicles = random.sample(cav_rl_agents, count)
	# 	else:
	# 		#consider all incoming edge vehicles including the edge on which the agent is plying
	# 		edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
	# 		all_vehicle_list = []
	# 		for v in cav_rl_agents:
	# 			all_vehicle_list.append(v)
	# 		for e_id in edge_list_incoming:   
	# 			all_vehicle = self.traci.edge.getLastStepVehicleIDs(e_id)
	# 			if len(all_vehicle)>0:
	# 				for v in all_vehicle:
	# 					priority_type = self.traci.vehicle.getTypeID(v)
	# 					if priority_type=="cav-priority" or priority_type=="rl-priority" or priority_type=="rl-default":
	# 						all_vehicle_list.append(v)
	# 		#randomly select 10 vehicles from this list
	# 		# all_vehicle_list = list(dict.fromkeys(all_vehicle_list))
	# 		all_vehicle_list.remove(agent_id)
	# 		count = min(len(all_vehicle_list),self._stateVehicleCount)

	# 		random_list_vehicles = random.sample(all_vehicle_list, count)

	# 	for veh in random_list_vehicles:
	# 		# the longitudinal position of the observed vehicle relative to the RL agent
	# 		pos = self.traci.vehicle.getPosition(veh)
	# 		agent_x = agent_pos[0]
	# 		agent_y = agent_pos[1]
	# 		x = pos[0]
	# 		y = pos[1]
	# 		rel_x =  agent_x - x
	# 		rel_y =  agent_y - y
	# 		hypo =  math.sqrt(rel_x**2 + rel_y**2)
	# 		theta = math.atan(rel_y/(rel_x+1e-5))
	# 		x_long =hypo*(math.cos(theta))
	# 		y_lat =hypo*(math.sin(theta))

	# 		speed = self.traci.vehicle.getSpeed(veh)
	# 		features.append(x_long/100)
	# 		features.append(y_lat/100)
	# 		features.append(speed/25)
	# 		priority_type = self.traci.vehicle.getTypeID(veh)
	# 		if priority_type=="cav-priority":
	# 			type = 0
	# 		elif priority_type=="rl-priority":
	# 			type = 1
	# 		elif priority_type=="rl-default":
	# 			type = 2
	# 		elif priority_type=="passenger-default":
	# 			type = 3
	# 		features.append(type/3)
	# 		# print(features)
		
	# 	#check if number of vehicle is less than 10

	# 	if len(features) < self._num_observation:
	# 		#missing number of features 
	# 		miss_n =  self._num_observation - len(features)
	# 		for iter in range(1, miss_n+1):
	# 			features.append(0)

	# 	state = features		

	# 	return np.array(state)
	
	def checkIfTeleport(self,agent_id):
		listt = self.traci.vehicle.getNextLinks(agent_id)
		if len(listt)==0:
			print("RL Agent does not exist. Adding dummy State for :" + str(agent_id))
			return True
		else:
			return False
		
	# def getState(self,agent_id):
	# 	"""
	# 	Retrieve the state of the network from sumo. 
	# 	"""
	# 	state = []
	# 	localRLVehicleCount = 0
	# 	allvehicleLocalcounter = 0
	# 	# state = [0,0,0,0,0,0,0,1,1]
	# 	# State = { Number of vehicle with priority lane access, number of vehicle without priority lane access, It’s own priority lane access, Avg. delay over all edges, number of emergency vehicle, number of public buses)
	# 	#Get the edgeID on which the RL agent is:
	# 	# agent_id = "RL_0"
	# 	# print(agent_id)
	# 	#Get the edgeID on which the RL agent is:
	# 	edge_id = self.traci.vehicle.getRoadID(agent_id)
	# 	lane_id = self.traci.vehicle.getLaneID(agent_id)			
	# 	priorityLane_id = edge_id + "_0"
	# 	# print(edge_id,agent_id)
	# 	# if edge_id == "":
	# 	# 	#load last state
	# 	# 	edge_id = self._lastEdge[agent_id]
	# 	# 	lane_id = self._lastLane[agent_id]
	# 	# 	nextLane = self._nextLane[agent_id]
	# 	# else:
	# 	# 	self._lastEdge[agent_id] = edge_id
	# 	# 	self._lastLane[agent_id] = lane_id

	# 	listt = self.traci.vehicle.getNextLinks(agent_id)
	# 	if self.checkIfTeleport(agent_id):
	# 		state = [0,0,0,0,0,0,0,0,0]			
	# 	else:
	# 		nextLane = listt[0][0]
	# 		self._nextLane[agent_id] = nextLane

	# 		routeList = self.traci.vehicle.getRoute(agent_id)
	# 		nextPriorityLane_id = nextLane.split("_")[0] + "_0"
	# 		#check occupancy of that lane
	# 		# occupancyNextLane = self.traci.lane.getLastStepOccupancy(nextLane)
	# 		# occupancyCurrentLane = self.traci.lane.getLastStepOccupancy(lane_id)
	# 		occupancyCurrentEdge = self.traci.edge.getLastStepOccupancy(edge_id)
	# 		# occupancyCurrentPriorityLane = self.traci.lane.getLastStepOccupancy(priorityLane_id)
	# 		occupancyNextPriorityLane = self.traci.lane.getLastStepOccupancy(nextPriorityLane_id)
	
	# 		nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
	# 		# now found the edges that is incoming to this junction
			
	# 		incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()

	# 		#count all  CAV vehicle behind this RL agent and within clearingThresholdDistance
	# 		agent_lane_pos = self.traci.vehicle.getLanePosition(agent_id)
	# 		agent_lane_pos_from_approaching_intersection = self.traci.lane.getLength(lane_id) - agent_lane_pos
	# 		all_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
	# 		edge_length = self.traci.lane.getLength(lane_id)
			
	# 		cavCount = 0
	# 		for cav in all_vehicle:
	# 			if cav !=agent_id:
	# 					allvehicleLocalcounter+=1
	# 			priority_type = self.traci.vehicle.getTypeID(cav)
	# 			if priority_type=="cav-priority":
	# 				cav_lane_position = self.traci.vehicle.getLanePosition(cav)
	# 				diff = agent_lane_pos - cav_lane_position
	# 				if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
	# 					cavCount+=1
	# 			elif priority_type=="rl-priority" or priority_type=="rl-default":
	# 				localRLVehicleCount+=1
		
	# 		#find next edge in the route for the RL agent

			
	# 		#traffic light phase related observations		
	# 		remainingDuration = 1
	# 		if priorityLane_id.find(":")!=-1 or lane_id.find(":")!=-1:
	# 			phaseState = 0
	# 		else:      
	# 			lanesList = self.traci.trafficlight.getControlledLanes(nextNodeID)
	# 			lanesListLink = self.traci.trafficlight.getControlledLinks(nextNodeID)
	# 			lanesListLink = list(filter(None, lanesListLink))
	# 			index = -1
	# 			indexOfConcern = 999
	# 			for element in lanesListLink:
	# 				index+=1
	# 				currentLane = element[0][0]
	# 				NextLane = element[0][1]
	# 				if currentLane == lane_id:
	# 					if NextLane == nextLane:
	# 						indexOfConcern = index
							

	# 			trafficState = self.traci.trafficlight.getRedYellowGreenState(nextNodeID)
	# 			phaseDuration = self.traci.trafficlight.getPhaseDuration(nextNodeID)
	# 			remainingPhaseDuration = self.traci.trafficlight.getNextSwitch(nextNodeID) - self.traci.simulation.getTime()
	# 			remainingDuration = remainingPhaseDuration/phaseDuration			
	# 			trafficStateList = list(trafficState)
	# 			# positionOf = lanesList.index(priorityLane_id)
	# 			if indexOfConcern!=999:
	# 				phase = trafficStateList[indexOfConcern]
	# 			else:
	# 				phase = "G"
	# 			phaseState = 0
	# 			if phase.find("G")!=-1 or phase.find("g")!=-1:
	# 				phaseState = 1
					
			
	# 		# edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
	# 		# occupancy = 0
	# 		# n_external_edges_counter = 0
	# 		# for e_id in edge_list_incoming: 
	# 		# 	#only add non-internal road occupancy
	# 		# 	if e_id.find(":") == -1:
	# 		# 		occupancy+= self.traci.edge.getLastStepOccupancy(e_id)
	# 		# 		n_external_edges_counter+=1
			
	# 		# avg_occ_all_incoming_lane = occupancy/n_external_edges_counter

	# 		# # print(edge_id)
	# 		# #Get the intersection the RL agent is going towards:
	# 		# # retrieve the successor edges of an edge
	# 		# nextEdges = self._net.getEdge(edge_id).getOutgoing()
	# 		# edge_list = [e.getID() for e in nextEdges] # list of all edges excluding internal
	# 		# nextIncomingEdges = self._net.getEdge(edge_id).getIncoming()
	# 		# edge_list_incoming = [e.getID() for e in nextIncomingEdges] # list of all edges excluding internal
	# 		priorityVehicleCount = 0
	# 		localPriorityRLVehicleCount = 0
	# 		localNonPriorityRLVehicleCount = 0
	# 		nonPriorityVehicleCount = 0
	# 		externalRLVehicleCount = 0
	# 		# total_waiting_time = 0
	# 		accumulated_time_loss = 0
	# 		localRLAgentList = []
	# 		normalization_totalNumberOfVehicle = 30
	# 		normalization_totalNumberOfCAV = 10
	# 		elapsed_simulation_time = self.traci.simulation.getTime()
	# 		allvehiclecounter = 0
	# 		edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
	# 		all_cav_count = 0
	# 		for e_id in edge_list_incoming: 
	# 			all_vehicle = self.traci.edge.getLastStepVehicleIDs(e_id)
	# 			for veh in all_vehicle:
	# 				if veh !=agent_id:
	# 					allvehiclecounter+=1
	# 					# elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)			
	# 					# accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh) / (elapsed_simulation_time - elapsed_vehicle_time)
	# 					# total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
	# 					priority_type = self.traci.vehicle.getTypeID(veh)	
	# 					lane_id = e_id + "_0"		
	# 					veh_position = self.traci.lane.getLength(lane_id) - self.traci.vehicle.getLanePosition(veh)	
	# 					if priority_type=="rl-priority" or priority_type=="rl-default":
	# 						if veh_position <= 40:
	# 							externalRLVehicleCount+=1
	# 					elif priority_type=="rl-default":
	# 						localNonPriorityRLVehicleCount+=1
	# 						localRLAgentList.append(veh)
	# 					elif priority_type=="cav-priority":
	# 						if e_id!=edge_id:
	# 							if veh_position<=40:
	# 								all_cav_count+=1
	# 				else:
	# 					nonPriorityVehicleCount+=1
	# 		self._listOfLocalRLAgents[agent_id] = localRLAgentList	
	# 		self._numberOfCAVApproachingIntersection[agent_id] = all_cav_count
	# 		elapsed_its_own_time = self.traci.vehicle.getDeparture(agent_id)	
	# 		# itsOwnTImeLoss = self.traci.vehicle.getTimeLoss(agent_id) / (elapsed_simulation_time - elapsed_its_own_time)
	# 		# self.lastTimeLoss[agent_id] = itsOwnTImeLoss
	# 		if self.traci.vehicle.getTypeID(agent_id)=="rl-priority":
	# 			itsPriorityAccess = 1			
	# 		else:
	# 			itsPriorityAccess = 0
	# 		# print(self._sumo_step)
	# 		# rlObs=0;cavObs=0
	# 		# if len(self._currentRLWaitingTimeForSpecificRLAgent)>0:
	# 		# 	if self._currentRLWaitingTimeForSpecificRLAgent[agent_id] - self._lastRLWaitingTimeForSpecificRLAgent[agent_id] > 0:
	# 		# 		rlObs = 0
	# 		# 	else:
	# 		# 		rlObs = 1
		
	# 		# 	if self._currentCAVWaitingTimeForSpecificRLAgent[agent_id] - self._lastCAVWaitingTimeForSpecificRLAgent[agent_id]>0:
	# 		# 		cavObs = 0
	# 		# 	else:
	# 		# 		cavObs = 1
	# 		# lane_index = int(lane_id.split("_")[1])
	# 		# if itsPriorityAccess==1:
	# 		# 	localPriorityRLVehicleCount=localPriorityRLVehicleCount-1
	# 		# else:
	# 		# 	localNonPriorityRLVehicleCount=localNonPriorityRLVehicleCount-1
	# 		# state = [itsOwnTImeLoss/self.action_steps,itsPriorityAccess,priorityVehicleCount/normalization_totalNumberOfVehicle,nonPriorityVehicleCount/normalization_totalNumberOfVehicle,accumulated_time_loss/self.action_steps,cavCount/normalization_totalNumberOfCAV,all_cav_count/normalization_totalNumberOfVehicle]
	# 		# state = [itsPriorityAccess,
	# 		#    (edge_length - agent_lane_pos)/edge_length, # distance from approaching intersection
	# 		#    lane_index/2,
	# 		#    occupancyCurrentLane,
	# 		#    occupancyCurrentPriorityLane,
	# 		#    occupancyNextLane,
	# 		#    occupancyNextPriorityLane,
	# 		#    localPriorityRLVehicleCount/allvehiclecounter,
	# 		#    localNonPriorityRLVehicleCount/allvehiclecounter,
	# 		#    nonPriorityVehicleCount/allvehiclecounter,
	# 		#    cavCount/allvehiclecounter,
	# 		#    all_cav_count/allvehiclecounter
	# 		#    ]
	# 		# state = [cavCount,all_cav_count,(edge_length - agent_lane_pos)/edge_length, occupancyCurrentLane,occupancyCurrentPriorityLane,occupancyNextLane,occupancyNextPriorityLane]
	# 		# total_rl_vehicle_atIntersection = localPriorityRLVehicleCount + 
	# 		if agent_lane_pos_from_approaching_intersection <=40 and phaseState==0: # pass state that includes approaching incoming vehicles
	# 			if allvehicleLocalcounter > 0:
	# 				state = [itsPriorityAccess,cavCount/allvehicleLocalcounter,all_cav_count/allvehiclecounter,externalRLVehicleCount/allvehiclecounter,(edge_length - agent_lane_pos)/edge_length,occupancyCurrentEdge,occupancyNextPriorityLane,phaseState,remainingDuration]
	# 			else:
	# 				if allvehiclecounter>0:
	# 					state = [itsPriorityAccess,cavCount,all_cav_count/allvehiclecounter,externalRLVehicleCount/allvehiclecounter,(edge_length - agent_lane_pos)/edge_length,occupancyCurrentEdge,occupancyNextPriorityLane,phaseState,remainingDuration]
	# 				else:
	# 					state = [itsPriorityAccess,cavCount,all_cav_count,externalRLVehicleCount,(edge_length - agent_lane_pos)/edge_length,occupancyCurrentEdge,occupancyNextPriorityLane,phaseState,remainingDuration]
	# 		else:
	# 			# pass states only for the lane on which agent is
	# 			if allvehicleLocalcounter > 0:
	# 				state = [itsPriorityAccess,cavCount/allvehicleLocalcounter,0,localRLVehicleCount/allvehicleLocalcounter,(edge_length - agent_lane_pos)/edge_length,occupancyCurrentEdge,0,0,0]
	# 			else:
	# 				state = [itsPriorityAccess,cavCount,0,localRLVehicleCount,(edge_length - agent_lane_pos)/edge_length,occupancyCurrentEdge,0,0,0]

	# 	# if allvehicleLocalcounter > 0:
	# 	# 	state = [itsPriorityAccess,cavCount/allvehicleLocalcounter]
	# 	# else:
	# 	# 	state = [itsPriorityAccess,cavCount]
	# 	# print("Count" + str(localPriorityRLVehicleCount))
          
	# 	# # if agent_id == "RL_1":
	# 	# # 	print(state)
	# 	return np.array(state)
	
	def getState(self,agent_id):
		"""
		Retrieve the state of the network from sumo. 
		"""
		state = []
		localRLVehicleCount = 0
		allvehicleLocalcounter = 1
		
		
		if self.checkIfTeleport(agent_id):
			state = [0,0,0,0,0,0,0]			
		else:
			listt = self.traci.vehicle.getNextLinks(agent_id)
			edge_id = self.traci.vehicle.getRoadID(agent_id)
			lane_id = self.traci.vehicle.getLaneID(agent_id)
			isPriorityLane = 1
			if lane_id.find("_0")==-1:
				isPriorityLane = 0

			nextLane = listt[0][0]
			self._nextLane[agent_id] = nextLane
			priorityLane_id = edge_id + "_0"
			routeList = self.traci.vehicle.getRoute(agent_id)
			nextPriorityLane_id = nextLane.split("_")[0] + "_0"
			#check occupancy of that lane
			# occupancyNextLane = self.traci.lane.getLastStepOccupancy(nextLane)
			# occupancyCurrentLane = self.traci.lane.getLastStepOccupancy(lane_id)
			# occupancyCurrentEdge = self.traci.edge.getLastStepOccupancy(edge_id)
			occupancyCurrentPriorityLane = self.traci.lane.getLastStepOccupancy(priorityLane_id)
			occupancyNextPriorityLane = self.traci.lane.getLastStepOccupancy(nextPriorityLane_id)
	
			nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
			# now found the edges that is incoming to this junction
			
			# incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()

			#count all  CAV vehicle behind this RL agent and within clearingThresholdDistance
			agent_lane_pos = self.traci.vehicle.getLanePosition(agent_id)
			# agent_lane_pos_from_approaching_intersection = self.traci.lane.getLength(lane_id) - agent_lane_pos
			all_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
			edge_length = self.traci.lane.getLength(lane_id)
			norm_agent_distance_from_intersection = (edge_length - agent_lane_pos)/edge_length
			
			cavCount = 0
			npcCount = 0
			for cav in all_vehicle:
				if cav !=agent_id:
						allvehicleLocalcounter+=1
				priority_type = self.traci.vehicle.getTypeID(cav)
				if priority_type=="cav-priority":					
					cav_lane_position = self.traci.vehicle.getLanePosition(cav)
					diff = agent_lane_pos - cav_lane_position
					if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
						cavCount+=1
				elif priority_type=="rl-priority" or priority_type=="rl-default":
					localRLVehicleCount+=1
				elif priority_type=="passenger-default" or priority_type=="rl-default":
					npc_lane_position = self.traci.vehicle.getLanePosition(cav)
					diff = agent_lane_pos - npc_lane_position
					if abs(diff)<= 1:
						npcCount+=1

		
			#find next edge in the route for the RL agent

			
			#traffic light phase related observations		
			remainingDuration = 1
			if priorityLane_id.find(":")!=-1 or lane_id.find(":")!=-1:
				phaseState = 0
			else:      
				lanesList = self.traci.trafficlight.getControlledLanes(nextNodeID)
				lanesListLink = self.traci.trafficlight.getControlledLinks(nextNodeID)
				lanesListLink = list(filter(None, lanesListLink))
				index = -1
				indexOfConcern = 999
				for element in lanesListLink:
					index+=1
					currentLane = element[0][0]
					NextLane = element[0][1]
					if currentLane == lane_id:
						if NextLane == nextLane:
							indexOfConcern = index
							

				trafficState = self.traci.trafficlight.getRedYellowGreenState(nextNodeID)
				phaseDuration = self.traci.trafficlight.getPhaseDuration(nextNodeID)
				remainingPhaseDuration = self.traci.trafficlight.getNextSwitch(nextNodeID) - self.traci.simulation.getTime()
				remainingDuration = remainingPhaseDuration/phaseDuration			
				trafficStateList = list(trafficState)
				# positionOf = lanesList.index(priorityLane_id)
				if indexOfConcern!=999:
					phase = trafficStateList[indexOfConcern]
				else:
					phase = "G"
				phaseState = 0
				if phase.find("G")!=-1 or phase.find("g")!=-1:
					phaseState = 1
			self._trafficPhaseRLagent[agent_id] = phaseState
			trafficState = self.traci.vehicle.getNextTLS(agent_id)
			isStopped = 0
			speed = self.traci.vehicle.getSpeed(agent_id)
		
			if speed == 0.0 and phaseState==0:
				isStopped = 1
			
			accumulated_time_loss = 0
			localRLAgentList = []
			
			
			self._listOfLocalRLAgents[agent_id] = localRLAgentList	
			if self.traci.vehicle.getTypeID(agent_id)=="rl-priority":
				itsPriorityAccess = 1			
			else:
				itsPriorityAccess = 0

			# state = [isPriorityLane,norm_agent_distance_from_intersection]
			state = [isPriorityLane,cavCount/allvehicleLocalcounter,speed/20,npcCount/allvehicleLocalcounter,remainingDuration,phaseState,norm_agent_distance_from_intersection]
			
		return np.array(state)
	

	def keepRLAgentLooping(self):
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle = self.getSplitVehiclesList(allVehicleList)
		missingRLAgentFlag = False
		# print(rl_vehicleID)
		if len(self._rl_vehicleID) != self.n: #this it solve: sometimes self.traci/sumo drops vehicle due to some reason. To maintain the same number of RL agent 
			# print("number of RL vehicle -",len(self._rl_vehicleID))
			missingRLAgentFlag = True #find missing vehicle ID	
			missingRLList = set(self.original_rl_vehicleID).difference(self._rl_vehicleID)
			print("RL agent missing : ",missingRLList)
		# for rl_veh in self._rl_vehicleID:
		# 	print("Inside 0")
		# 	if self.traci.vehicle.getRouteIndex(str(rl_veh)) == (len(self.traci.vehicle.getRoute(str(rl_veh))) - 1): #Check to see if the car is at the end of its route
		# 		new_destiny = random.choice(self._releventEdgeId)
		# 		# print(str(rl_veh)+str(new_destiny))
		# 		self.traci.vehicle.changeTarget(str(rl_veh),str(new_destiny)) #Assign random destination
		if missingRLAgentFlag:
			# print(self.original_rl_vehicleID)
			# print(self._rl_vehicleID)
			for missRL in missingRLList:
				if missRL not in self.traci.simulation.getArrivedIDList():
					# print(self.traci.vehicle.getIDList())
					# try:
					# 	self.traci.vehicle.remove(missRL) # just in case
					# except:
					# 	pass # do nothing					
					edges = self._routeDict[missRL]
					# print(edges)
					if missRL not in self.traci.simulation.getArrivedIDList():
						#create the route
						# print(self.traci.route.getIDList())
						try:
							self.traci.route.add(missRL,edges)					
						except:
							pass
					# print(self.traci.route.getEdges(missRL))
					
					# try:					
					self.traci.simulationStep() 
					self.traci.vehicle.add(missRL,missRL,typeID="rl-priority")
					# except:
					# 	pass #do nothing
	
	def resetAllVariables(self):
		self.lastActionDict.clear()
		self._timeLossOriginalDict.clear()
		self._currentCAVWaitingTimeForSpecificRLAgent.clear()
		self._lastCAVWaitingTimeForSpecificRLAgent.clear()
		self._currentRLWaitingTimeForSpecificRLAgent.clear()
		self._lastRLWaitingTimeForSpecificRLAgent.clear()
		self._listOfLocalRLAgents.clear()
		self._currentTimeLoss_rl = 0
		self._currentTimeLoss_npc = 0
		self._currentTimeLoss_cav = 0
		self._currentTimeLoss_Heuristic = 0
		self._avg_speed_rl = 0
		self._avg_speed_heuristic = 0
		self._avg_speed_npc = 0
		self._avg_speed_cav = 0
		self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0
		self._average_edge_occupancy = 0
		self._average_priorityLane_occupancy = 0
		self._average_throughput = 0
		self._average_PMx_emission = 0
		self._average_LaneChange_number = 0
		self._average_LaneChange_number_all = 0
		self._average_LaneChange_number_rl = 0
		self._episodeStep = 0	
		self._average_throughput = 0
		self._collisionCounter = 0
		self._rl_counter = 0
		self._cavFlowCounter=0
		self._collisionCount = 0

		# ## ADD RL AGENTS DYNAMICALLY
		# for veh in self.controlled_vehicles:
		# 	veh_id = veh.name
		# 	self.traci.vehicle.addFull(veh_id, '', typeID='rl-priority', depart=0)
		# 	init_edge, = self.traci.vehicle.getRoute(veh_id) # a random edge was assigned
		# 	route = np.random.choice(self._allEdgeIds, size=100, replace=True).tolist()
		# 	route = [init_edge] + route
		# 	self.traci.vehicle.setVia(veh_id, route)
		# 	self.traci.vehicle.rerouteTraveltime(veh_id)
		# 	self.traci.vehicle.moveTo(veh_id, f'{route[0]}_0', 0)

	def initializeRLAgentStartValues(self):
		for rl_agent in self.original_rl_vehicleID:
			self._timeLossOriginalDict[rl_agent] = self.traci.vehicle.getTimeLoss(rl_agent) # store the time loss when the agent is spawned. It will be used to compare for action step time loss for reward calculation
			# self._travelTime[rl_agent] = 
	
	def setHeuristicAgentTogglePriority(self): #heuristic logic to change priority of all agent starting with heuristic in the name. 
		#this is done to overcome the limitation of training 100's of RL agent. Can we just train 25% of the RL agent with heuristic logic and 
		#still get similar or better training output? One novelty of the paper, probably?
		allVehicleList = self.traci.vehicle.getIDList()
		# print("Total number of vehicles",len(allVehicleList))
		self._npc_vehicleID,self._rl_vehicleID, self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)

		counterDefault = 0
		counterPriority = 0
		for heuristic in self._heuristic_vehicleID:
			which_lane = self.traci.vehicle.getLaneID(heuristic)
			if self.edgeIdInternal(which_lane)==False:
				lane_index = which_lane.split("_")[1]
				which_edge = which_lane.split("_")[0]
				priority_lane = which_edge + str("_0") # find priority lane for that vehicle
				vehicle_on_priority_lane = self.traci.lane.getLastStepVehicleIDs(priority_lane)
				npc_vehicleID,rl_vehicleID, heuristic_vehicleID,cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(vehicle_on_priority_lane)
				heuristic_lane_position = self.traci.vehicle.getLanePosition(heuristic)
				
				flag = False
				diff = 0
				cav_lane_position = -999
				for cav in cav_vehicleID:
					cav_lane_position = self.traci.vehicle.getLanePosition(cav)
					diff = heuristic_lane_position - cav_lane_position
					if diff>= self._lane_clearing_distance_threshold and diff>0:					
						continue
					else:
						#change priority of heuristic agent as it is inside clearing distance
						# speed = self.traci.vehicle.getSpeed(heuristic)
						if diff > 0:
							# print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
							self.traci.vehicle.setType(heuristic,"heuristic-default")
							flag = True
							counterDefault+=1
							# 	self.traci.vehicle.changeLane(heuristic,0,self._laneChangeAttemptDuration) 
							# else:
							# 	self.traci.vehicle.changeLane(heuristic,1, self._laneChangeAttemptDuration)
							break
				if flag==False: 
					if lane_index!=0 and self.traci.vehicle.getTypeID(heuristic)!="heuristic-priority":
						# speed = self.traci.vehicle.getSpeed(heuristic)
						# if speed>0.2:
						self.traci.vehicle.setType(heuristic,"heuristic-priority")
						counterPriority+=1
						# if cav_lane_position ==-999:
						# 	print(str(diff),"--",str(heuristic_lane_position),"--","No CAV present")
						# else:
						# 	print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
						# self.traci.vehicle.changeLane(heuristic,2,50) 
					

		# print("PtoD change =",counterDefault,"  DtoP changes =",counterPriority)

	def setRLAgentTogglePriority(self): #heuristic logic to change priority of all agent starting with heuristic in the name. 
		#this is done to overcome the limitation of training 100's of RL agent. Can we just train 25% of the RL agent with heuristic logic and 
		#still get similar or better training output? One novelty of the paper, probably?
		allVehicleList = self.traci.vehicle.getIDList()
		# print("Total number of vehicles",len(allVehicleList))
		self._npc_vehicleID,self._rl_vehicleID, self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)

		counterDefault = 0
		counterPriority = 0
		for rl in self._rl_vehicleID:
			which_lane = self.traci.vehicle.getLaneID(rl)
			npcCount = 0
			if self.edgeIdInternal(which_lane)==False:
				lane_index = which_lane.split("_")[1]
				which_edge = which_lane.split("_")[0]
				priority_lane = which_edge + str("_0") # find priority lane for that vehicle
				lane_01 = which_edge + str("_1") # find priority lane for that vehicle
				vehicle_on_priority_lane = self.traci.lane.getLastStepVehicleIDs(priority_lane)
				npc_vehicleID,rl_vehicleID, heuristic_vehicleID,cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(vehicle_on_priority_lane)
				heuristic_lane_position = self.traci.vehicle.getLanePosition(rl)
				vehicle_on_lane = self.traci.lane.getLastStepVehicleIDs(lane_01)
				for veh in vehicle_on_lane:
					npc_lane_position = self.traci.vehicle.getLanePosition(veh)
					diff = heuristic_lane_position - npc_lane_position
					if abs(diff)<= 2.5:
						npcCount+=1
				
				flag = False
				diff = 0
				cav_lane_position = -999
				for cav in cav_vehicleID:
					cav_lane_position = self.traci.vehicle.getLanePosition(cav)
					diff = heuristic_lane_position - cav_lane_position
					if diff>= self._lane_clearing_distance_threshold:					
						continue
					else:
						#change priority of heuristic agent as it is inside clearing distance
						# speed = self.traci.vehicle.getSpeed(heuristic)
						if diff > 0 and npcCount==0:
						# if diff > 0:
							# print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
							self._collisionCounter+=1
							self.traci.vehicle.setType(rl,"rl-default")
							lane_index = self.traci.vehicle.getLaneIndex(rl)
							if lane_index==0:
								self.traci.vehicle.changeLane(rl,1,0)
							flag = True
							break
				if flag==False: 
					# if lane_index!=0 and self.traci.vehicle.getTypeID(rl)!="rl-priority":
					if lane_index!=0:
						# speed = self.traci.vehicle.getSpeed(heuristic)
						# if speed>0.2:
						self.traci.vehicle.setType(rl,"rl-priority")
						lane_index = self.traci.vehicle.getLaneIndex(rl)
						if lane_index!=0:
							self.traci.vehicle.changeLane(rl,0,0)
							
						self._collisionCounter+=1
						# counterPriority+=1
						# if cav_lane_position ==-999:
						# 	print(str(diff),"--",str(heuristic_lane_position),"--","No CAV present")
						# else:
						# 	print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
						# self.traci.vehicle.changeLane(heuristic,2,50) 
					

		# print("PtoD change =",counterDefault,"  DtoP changes =",counterPriority)

	def reset(self):		
		print("--------Inside RESET---------")
		
		print("Collision_Counter :" + str(self._collisionCount))
		self._sumo_step = 0
		obs_n = []
		seed = self._sumo_seed
		self.traci.load(self.sumoCMD + ["--seed", str(seed)])
		self.resetAllVariables()

		#WARMUP PERIOD
		while self._sumo_step <= self._warmup_steps:
			self.traci.simulationStep() 		# Take a simulation step to initialize	
			# print(self.traci.vehicle.getTimeLoss("RL_9"))
			# if self._sumo_step == 10:
			# 	allVehicleList = self.traci.vehicle.getIDList()
			# 	self._npc_vehicleID,self.original_rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
				
				# self.initializeRLAgentStartValues()
			# 	self.keepRLAgentLooping()
			# if self._sumo_step%self.action_steps==0:
			# 	# self.setHeuristicAgentTogglePriority()
			self.setRLAgentTogglePriority()
			self._sumo_step +=1

		#record observatinos for each agent
		
		# self.initializeNPCRandomPriority()
		for agent in self.agents:
			agent.done = False
			obs_n.append(self._get_obs(agent))
		#keep list of all route ID
		# if self._alreadyAddedFlag == False:
		# 	for rl_veh in self.original_rl_vehicleID:
		# 		self.traci.route.add(rl_veh,self.traci.vehicle.getRoute(rl_veh))
		# 		self._routeDict[rl_veh] = self.traci.vehicle.getRoute(rl_veh)
		# 		self._alreadyAddedFlag = True
		# # 		# print(rl_veh,self.traci.vehicle.getRoute(rl_veh))
		print("--------Outside RESET---------" + str(self._sumo_step))
		return obs_n

	# get observation for a particular agent
	def _get_obs(self, agent):
		return self.getState(f'RL_{agent.id}')
		# state = [0.1,0.3,0.5,0.6,0.8]
		# return state

	def computeCooperativeReward(self,rl_agent):
		elapsed_simulation_time = self.traci.simulation.getTime()
		elapsed_its_own_time = self.traci.vehicle.getDeparture(rl_agent)
		currentTimeLoss = self.traci.vehicle.getTimeLoss(rl_agent) / (elapsed_simulation_time - elapsed_its_own_time)
		diffTimeLoss = self.lastTimeLossRLAgents[rl_agent] - currentTimeLoss
		
		# if diffTimeLoss > 0 : It means good for positive reward
		diffTimeLossInSeconds = diffTimeLoss*(elapsed_simulation_time - elapsed_its_own_time)
		# if agent_id == "RL_0":
		# 	print(diffTimeLossInSeconds)
		# 	print(self.lastTimeLossRLAgents[agent_id]*(elapsed_simulation_time - elapsed_its_own_time))
		# 	print(currentTimeLoss*(elapsed_simulation_time - elapsed_its_own_time))

		if self.traci.vehicle.getTypeID(rl_agent)=="rl-priority": #check if agent 
			if self.lastActionDict[rl_agent] == 0: # give up the priority
				reward = +1*(diffTimeLoss)
				# if diffTimeLoss > 0:	
				# 	reward = +1*(diffTimeLoss)
				# else:
				# 	reward = +1*(diffTimeLoss)
				
			elif self.lastActionDict[rl_agent] == 1: # do nothing. Keep the same action
				reward = +1*(diffTimeLoss)
				# if diffTimeLoss > 0:				
				# 	reward = -1*(diffTimeLoss) #penalize because it is not cooperative even if it can
				# else:
				# 	reward = +1*(diffTimeLoss) #reward because it is trying to maximize it's own gain
			else: # ask for priority if priority				
				if diffTimeLoss > 0:
					reward = -0.1
				else:
					reward = 0

		if self.traci.vehicle.getTypeID(rl_agent)=="rl-default": #check if agent 
			if self.lastActionDict[rl_agent] == 0: # give up the priority
				if diffTimeLoss > 0:
					reward = 0
				else:
					reward = -0.1
			elif self.lastActionDict[rl_agent] == 1: # do nothing. Keep the same action
				reward = +1*(diffTimeLoss)
			else: # ask for priority if no priority
				reward = -1*(diffTimeLoss)

		return reward
	
	def computeOverallNetworkReward(self,rl_agent):		
		# delta_time_loss = self._currentOverAllTimeLoss[rl_agent] - self._lastOverAllTimeLoss[rl_agent]
		delta_waiting_time_loss = self._currentOverAllWaitingTime[rl_agent] - self._lastOverAllWaitingTime[rl_agent]
		# print(delta_time_loss,"--", self._currentOverAllWaitingTime[rl_agent], "--", self._lastOverAllWaitingTime[rl_agent])
		if delta_waiting_time_loss > 0:
			reward_timeLoss = -1
		else:
			reward_timeLoss = +1
		
		return reward_timeLoss

	def computeCAVReward(self,rl_agent):
		before_priority = self._beforePriorityForRLAgent[rl_agent]
		after_priority = self._afterPriorityForRLAgent[rl_agent]
		which_lane = self.traci.vehicle.getLaneID(rl_agent)		
		# bestLanesTuple = self.traci.vehicle.getBestLanes(rl_agent)
  
		if self._numberOfCAVWithinClearingDistanceAfter[rl_agent] == 0 and self._numberOfCAVWithinClearingDistanceBefore[rl_agent] > 0 and which_lane.find("_0")!=-1:
			reward = +0.5
		elif self._numberOfCAVWithinClearingDistanceAfter[rl_agent] == 0 and self._numberOfCAVWithinClearingDistanceBefore[rl_agent] == 0 and self._numberOfCAVWithinClearingDistanceOnPLAfter[rl_agent]>0:
			reward = +0.5
		elif self._numberOfCAVWithinClearingDistanceAfter[rl_agent] > 0 and self._numberOfCAVWithinClearingDistanceBefore[rl_agent] == 0:
			reward = -0.5
		elif self._numberOfCAVWithinClearingDistanceAfter[rl_agent] > 0:
			reward = -0.5
		elif self._numberOfCAVWithinClearingDistanceAfter[rl_agent] == 0 and self._numberOfCAVWithinClearingDistanceBefore[rl_agent] == 0 and self._numberOfCAVWithinClearingDistanceOnPLAfter[rl_agent]==0 and which_lane.find("_0")!=-1:
			reward = +0.5
		# elif before_priority =="rl-default" and after_priority=="rl-priority" and self._numberOfCAVWithinClearingDistanceOnPLAfter[rl_agent]==0:
		# 	reward = +0.5
		# elif before_priority =="rl-priority" and after_priority=="rl-priority" and self._numberOfCAVWithinClearingDistanceOnPLAfter[rl_agent]==0:
		# 	reward = +0.5
		else:
			reward = 0

		
		return reward

	def computePriorityLaneThroughput(self,rl_agent):
		reward = +0.5
		if self._throughputBefore[rl_agent] > self._throughputAfter[rl_agent]:
			reward = -0.5
		elif self._throughputBefore[rl_agent] == self._throughputAfter[rl_agent]:
			reward = 0
		elif self._throughputBefore[rl_agent] == 0 and self._throughputAfter[rl_agent]==0:
			reward = 0
		
		return reward
	
	def computeRLRewardDistFromIntersection(self,rl_agent):
		reward = 0
		agent_lane_pos = self.traci.vehicle.getLanePosition(rl_agent)
		lane_id = self.traci.vehicle.getLaneID(rl_agent)	
		edge_length = self.traci.lane.getLength(lane_id)
		agent_distance_from_intersection = (edge_length - agent_lane_pos)
		if agent_distance_from_intersection < 50:
			speed = self.traci.vehicle.getSpeed(rl_agent)
			phase = self._trafficPhaseRLagent[rl_agent]
			reward = +0.5		
			if lane_id.find("_0")!=-1 and phase!=1:
				# all_vehicle = self.traci.lane.getLastStepVehicleIDs(lane_id)			
				# cavCount = 0
				# for cav in all_vehicle:
				# 	priority_type = self.traci.vehicle.getTypeID(cav)
				# 	if priority_type=="cav-priority":					
				# 		cav_lane_position = self.traci.vehicle.getLanePosition(cav)
				# 		diff = agent_lane_pos - cav_lane_position
				# 		if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
				# 			cavCount+=1
				# #penalize it with small negative reward as it is stuck infront of the red light
				# if cavCount > 0:
					reward = -0.5				
		return reward

		
	def computeCAVAccumulatedWaitingTime(self,rl_agent):
		if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] >0 and self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]> 0:			
			if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]> 0:
			# cav_delay = -(self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]
				# cav_delay = -(self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]   
				cav_delay = -0.5
			else:
				if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]!=self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]:
					# cav_delay = +(self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]
					cav_delay = +0.5
				else:
					cav_delay=0
		else:
			if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] == 0: 
				cav_delay = +0.5
			else:
				cav_delay = -0.5

		# cav_delay = (self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent])/25
		# print(cav_delay)
		return cav_delay
	
	def computeRLAccumulatedWaitingTime(self,rl_agent):
		if self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] >0 and self._lastRLWaitingTimeForSpecificRLAgent[rl_agent]> 0:			
			if self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] - self._lastRLWaitingTimeForSpecificRLAgent[rl_agent]> 0:
			# cav_delay = -(self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]
				rl_delay = -(self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] - self._lastRLWaitingTimeForSpecificRLAgent[rl_agent])/self._currentRLWaitingTimeForSpecificRLAgent[rl_agent]   
			else:
				if self._currentRLWaitingTimeForSpecificRLAgent[rl_agent]!=self._lastRLWaitingTimeForSpecificRLAgent[rl_agent]:
					rl_delay = +(self._lastRLWaitingTimeForSpecificRLAgent[rl_agent] - self._currentRLWaitingTimeForSpecificRLAgent[rl_agent])/self._lastRLWaitingTimeForSpecificRLAgent[rl_agent]
				else:
					rl_delay=0
		else:
			if self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] == 0: 
				rl_delay = +0.5
			else:
				rl_delay = -0.5
  	
		return rl_delay

	def computeAvgSpeedPriorityLaneReward(self,rl_agent):
		reward = 0
		all_LD = self.traci.inductionloop.getIDList()
		avg_Speed = 0
		rl_agent_lane = self.traci.vehicle.getLaneID(rl_agent)
		max_Speed = self.traci.lane.getMaxSpeed(rl_agent_lane)
		for ld in all_LD:
			avg_Speed += self.traci.inductionloop.getLastIntervalMeanSpeed(ld)
		avg_Speed = avg_Speed/len(all_LD)
		reward = avg_Speed/max_Speed

		return reward
		
	def computeRLSpeedReward(self,rl_agent):
		reward = 0
		beforeSpeed = self._BeforeSpeed[rl_agent]
		afterSpeed = self._AfterSpeed[rl_agent]
		if afterSpeed >=beforeSpeed:
			reward = +0.5
		else:
			reward = -0.5
		return reward

	def safetyReward(self,rl_agent):
		reward = 0.5
		if rl_agent in self._collisionVehicleID:
			reward = -0.5
		return reward
	
	def computeCAVSpeedReward(self,rl_agent):
		reward = 0
		beforeSpeed = self._BeforeCAVSpeed[rl_agent]
		afterSpeed = self._AfterCAVSpeed[rl_agent]
		if afterSpeed >=beforeSpeed:
			reward = +0.5
		else:
			reward = -0.5
		return reward

	# get reward for a particular agent
	def _get_reward(self,agent):
		agent_id = f'RL_{agent.id}'
		overall_reward = 0
		if self._isTestFlag == False:
			if len(self.lastActionDict) !=0:				
				# reward_cooperative = self.computeCooperativeReward(agent_id)
				# reward_overallNetwork = self.computeOverallNetworkReward(agent_id)
				# reward_cavWaitingTime = self.computeCAVAccumulatedWaitingTime(agent_id)
				# reward_RLWaitingTime = self.computeRLAccumulatedWaitingTime(agent_id)
				reward_speed_RL = self.computeRLSpeedReward(agent_id)
				reward_speed_CAV = self.computeCAVSpeedReward(agent_id)
				# reward_safety = self.safetyReward(agent_id)
				# reward_priorityLane_Throughput = self.computePriorityLaneThroughput(agent_id)
				reward_cav_priority = self.computeCAVReward(agent_id)
				reward_dist_intersection = self.computeRLRewardDistFromIntersection(agent_id)
				
				# print("reward: " + str(reward_cav_priority))
				# print("-------------------------")
				# reward_priority_lane_Speed = self.computeAvgSpeedPriorityLaneReward(agent_id)
				# overall_reward = reward_cooperative + reward_overallNetwork + reward_cav_priority
				# overall_reward = reward_cav_priority
				# print(overall_reward)		
				# overall_reward = self._weightCAVPriority*reward_cav_priority + self._weightRLWeightingTime*reward_RLWaitingTime + self._weightCAVWeightingTime*reward_cavWaitingTime
				# overall_reward = reward_cav_priority + 0.2*reward_RLWaitingTime + 0.2*reward_cavWaitingTime
				overall_reward = reward_dist_intersection + reward_cav_priority + reward_speed_RL + reward_speed_CAV
				# overall_reward = reward_cav_priority
		return overall_reward
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _get_done(self, agent):
		if self.traci.simulation.getTime() >= self.episode_duration and self._isTestFlag==False:
			return True
		return agent.done

	# get info used for benchmarking
	def _get_info(self, agent):
		return {}
	def edgeIdInternal(self,edge_id):
		if edge_id.find(":") == -1:
			return False
		else:
			return True
	
	def set_sumo_seed(self, seed):
		self._sumo_seed = seed


	def set_Testing(self,flag):
		self._isTestFlag = flag
    
	def getSplitVehiclesList(self,allvehicles):
		rl_vehicleID = []
		cav_vehicleID = []
		heuristic_vehicleID = []
		npc_vehicleID = []
		haltVehicleCount=0
		ratioOfHaltVehicle=0
		total_vehicle_count = len(allvehicles)
		for veh in allvehicles:
			speed = self.traci.vehicle.getSpeed(veh)
			if speed < 0.11:
				haltVehicleCount+=1
			x = veh.split("_",1)
			if x[0] =="RL":
				rl_vehicleID.append(veh)
			elif x[0] == "cav":
				cav_vehicleID.append(veh)
			elif x[0] == "heuristic":
				heuristic_vehicleID.append(veh)
			elif x[0] == "npc":
				npc_vehicleID.append(veh)
		if total_vehicle_count>0:
			ratioOfHaltVehicle = haltVehicleCount/total_vehicle_count
		return npc_vehicleID,rl_vehicleID,heuristic_vehicleID,cav_vehicleID,ratioOfHaltVehicle

	# def collectObservation(self,lastTimeStepFlag):
	# 	#This function collects sum of time loss for all vehicles related to a in-concern RL agent. 
		
	# 	allVehicleList = self.traci.vehicle.getIDList()
	# 	self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
	# 	elapsed_simulation_time = self.traci.simulation.getTime()
	# 	if lastTimeStepFlag:
	# 		# for edge_id in self.edge_list:
	# 		# 	priority_lane = edge_id + "_0"
	# 		# 	occupancy = self.traci.edge.getLa
			
	# 		self._listOfVehicleIdsInConcern.clear()
	# 		for rl_agent in self._rl_vehicleID:
	# 			elapsed_its_own_time = self.traci.vehicle.getDeparture(rl_agent)
	# 			itsOwnTImeLoss = self.traci.vehicle.getTimeLoss(rl_agent) / (elapsed_simulation_time - elapsed_its_own_time)
	# 			self.lastTimeLossRLAgents[rl_agent] = itsOwnTImeLoss
	# 			edge_id = self.traci.vehicle.getRoadID(rl_agent)
	# 			self._beforePriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
	# 			agent_lane_pos = self.traci.vehicle.getLanePosition(rl_agent)
		
	# 			# edge_id = self.traci.vehicle.getRoadID(rl_agent)
	# 			lane_id = self.traci.vehicle.getLaneID(rl_agent)

	# 			# inductionloop_id = "det_" + str(lane_id.split("_")[0]) + "_0_1_passenger"
				
	# 			# throughput = self.traci.inductionloop.getLastIntervalVehicleNumber(inductionloop_id)
	# 			# self._throughputBefore[rl_agent] = throughput
	# 			#check if agent is on a priority lane. Count only those CAV's for it. RL agent on other lanes will have CAV count as zero
	# 			# lane_id = self.traci.vehicle.getLaneID(rl_agent)
	# 			cavCount = 0
	# 			# if lane_id.split("_")[1] == "0":
	# 			all_cav_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
	# 			for cav in all_cav_vehicle:
	# 				cav_lane_position = self.traci.vehicle.getLanePosition(cav)
	# 				diff = agent_lane_pos - cav_lane_position
	# 				if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
	# 					cavCount+=1

	# 			if lane_id.split("_")[1] == "0":
	# 				self._numberOfCAVWithinClearingDistanceBefore[rl_agent] = cavCount
	# 			else:
	# 				self._numberOfCAVWithinClearingDistanceBefore[rl_agent] = 0

   
		
	# 			accumulated_time_loss = 0
	# 			total_waiting_time=0
	# 			total_waiting_time_cav = 0
	# 			# print(edge_id,agent_id)
	# 			#check if edge_id is internal
	# 			# if self.edgeIdInternal(edge_id):
	# 			# 	print("Internal Edge ID - ",edge_id) #change it to the main edge_id
	# 			vehicle_list =[]
	# 			nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
	# 			# now found the edges that is incoming to this junction
	# 			incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()
	# 			edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
	# 			for e_id in edge_list_incoming:   
	# 				all_vehicle = self.traci.edge.getLastStepVehicleIDs(e_id)
	# 				if len(all_vehicle)>0:
	# 					vehicle_list+=all_vehicle
	# 				for veh in all_vehicle:
	# 					elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)
	# 					accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh)/(elapsed_simulation_time - elapsed_vehicle_time)
	# 					total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
	# 					priority_type = self.traci.vehicle.getTypeID(veh)
	# 					if priority_type=="cav-priority": 
	# 						total_waiting_time_cav+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
				
	# 			self._listOfVehicleIdsInConcern[rl_agent] = vehicle_list
	# 			self._lastRLWaitingTimeForSpecificRLAgent[rl_agent] = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
	# 			# print("Before Waiting Time for RL = ",self._lastRLWaitingTimeForSpecificRLAgent[rl_agent])
	# 			# print("Before Waiting Time for CAV = ",total_waiting_time_cav)
				
	# 			self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time_cav				
	# 			self._lastOverAllTimeLoss[rl_agent] = accumulated_time_loss
	# 			self._lastOverAllWaitingTime[rl_agent] = total_waiting_time
	# 	else:
			
	# 		for rl_agent in self._rl_vehicleID:	
	# 			accumulated_time_loss = 0
	# 			total_waiting_time=0
	# 			total_waiting_time_cav=0
	# 			self._afterPriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
	# 			for veh in self._listOfVehicleIdsInConcern[rl_agent]:
	# 				if veh in allVehicleList:
	# 					elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)
	# 					accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh)/(elapsed_simulation_time - elapsed_vehicle_time)
	# 					total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
	# 					priority_type = self.traci.vehicle.getTypeID(veh)
	# 					if priority_type=="cav-priority": 
	# 						total_waiting_time_cav+=self.traci.vehicle.getAccumulatedWaitingTime(veh)

	# 			lane_id = self.traci.vehicle.getLaneID(rl_agent)
	# 			# inductionloop_id = "det_" + str(lane_id.split("_")[0]) + "_0_1_passenger"
	# 			# throughput = self.traci.inductionloop.getLastIntervalVehicleNumber(inductionloop_id)
	# 			# self._throughputAfter[rl_agent] = throughput
    
	# 			self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
	# 			# print("Current Waiting Time for RL = ",self._currentRLWaitingTimeForSpecificRLAgent[rl_agent])
	# 			# print("Current Waiting Time for CAV = ",total_waiting_time_cav)
	# 			self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time_cav
	# 			self._currentOverAllWaitingTime[rl_agent] = total_waiting_time
    
	# 			agent_lane_pos = self.traci.vehicle.getLanePosition(rl_agent)
    
	# 			edge_id = self.traci.vehicle.getRoadID(rl_agent)
	# 			#check if agent is on a priority lane. Count only those CAV's for it. RL agent on other lanes will have CAV count as zero
	# 			# lane_id = self.traci.vehicle.getLaneID(rl_agent)
	# 			cavCount = 0
	# 			# if lane_id.split("_")[1] == "0":
	# 			all_cav_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
	# 			for cav in all_cav_vehicle:
	# 				cav_lane_position = self.traci.vehicle.getLanePosition(cav)
	# 				diff = agent_lane_pos - cav_lane_position
	# 				if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
	# 					cavCount+=1
	# 			self._numberOfCAVWithinClearingDistanceOnPLAfter[rl_agent] = cavCount
	# 			if lane_id.split("_")[1] == "0":
	# 				self._numberOfCAVWithinClearingDistanceAfter[rl_agent] = cavCount
	# 			else:
	# 				self._numberOfCAVWithinClearingDistanceAfter[rl_agent] = 0

	def collectObservation(self,lastTimeStepFlag):
		#This function collects sum of time loss for all vehicles related to a in-concern RL agent. 
		
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
		elapsed_simulation_time = self.traci.simulation.getTime()
		
		if lastTimeStepFlag:
			# self._listOfVehicleIdsInConcern.clear()
			for rl_agent in self._rl_vehicleID:
				if self.checkIfTeleport(rl_agent):
					self._listOfVehicleIdsInConcern[rl_agent] = []
					continue

				edge_id = self.traci.vehicle.getRoadID(rl_agent)
				self._beforePriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
				agent_lane_pos = self.traci.vehicle.getLanePosition(rl_agent)
		
				# edge_id = self.traci.vehicle.getRoadID(rl_agent)
				lane_id = self.traci.vehicle.getLaneID(rl_agent)

				self._BeforeSpeed[rl_agent] = self.traci.vehicle.getSpeed(rl_agent)
				# inductionloop_id = "det_" + str(lane_id.split("_")[0]) + "_0_1_passenger"
				
				# throughput = self.traci.inductionloop.getLastIntervalVehicleNumber(inductionloop_id)
				# self._throughputBefore[rl_agent] = throughput
				#check if agent is on a priority lane. Count only those CAV's for it. RL agent on other lanes will have CAV count as zero
				# lane_id = self.traci.vehicle.getLaneID(rl_agent)
				vehicle_list =[]
				cavCount = 0
				# if lane_id.split("_")[1] == "0":
				all_cav_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
				
				total_waiting_time_cav = 0
				total_cav_speed = 0
				for cav in all_cav_vehicle:
					priority_type = self.traci.vehicle.getTypeID(cav)
					if priority_type=="cav-priority": 		
						cav_lane_position = self.traci.vehicle.getLanePosition(cav)
						diff = agent_lane_pos - cav_lane_position
						if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
							cavCount+=1							
							vehicle_list.append(cav)
							speed = self.traci.vehicle.getSpeed(cav)
							if speed == 0.0:
								g = 0
								edge_length = self.traci.lane.getLength(lane_id)
								distanceFromIntersection = edge_length - cav_lane_position
							# if self.traci.vehicle.isStopped(cav)==True:
							# 	g = 0
							# 	edge_length = self.traci.lane.getLength(lane_id)
							# 	distanceFromIntersection = edge_length - cav_lane_position
							cav_wait_time = self.traci.vehicle.getAccumulatedWaitingTime(cav)
							# if cav_wait_time > 2:
							# 	g = 0
							total_waiting_time_cav+=cav_wait_time
							# total_waiting_time_cav+=self.traci.vehicle.getAccumulatedWaitingTime(cav)
							total_cav_speed+= self.traci.vehicle.getSpeed(cav)
				# print("Before: CAV Waiting Time --" + str(total_cav_speed))
				if lane_id.split("_")[1] == "0":
					self._numberOfCAVWithinClearingDistanceBefore[rl_agent] = cavCount
				else:
					self._numberOfCAVWithinClearingDistanceBefore[rl_agent] = 0


				
				self._listOfVehicleIdsInConcern[rl_agent] = vehicle_list
				self._lastRLWaitingTimeForSpecificRLAgent[rl_agent] = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
				self._BeforeCAVSpeed[rl_agent] = total_cav_speed
				self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time_cav	
		else:			
			for rl_agent in self._rl_vehicleID:
				if self.checkIfTeleport(rl_agent):
					continue
				accumulated_time_loss = 0
				total_waiting_time=0
				total_waiting_time_cav=0
				total_cav_speed = 0
				self._AfterSpeed[rl_agent] = self.traci.vehicle.getSpeed(rl_agent)
				self._afterPriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
				lane_id = self.traci.vehicle.getLaneID(rl_agent)
				for veh in self._listOfVehicleIdsInConcern.get(rl_agent, []):
					if veh in allVehicleList:
						# elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)
						# accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh)/(elapsed_simulation_time - elapsed_vehicle_time)
						total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
						priority_type = self.traci.vehicle.getTypeID(veh)
						if priority_type=="cav-priority": 
							cav_lane_position = self.traci.vehicle.getLanePosition(veh)
							# if self.traci.vehicle.isStopped(veh)==True:
							# 	g = 0
							# 	edge_length = self.traci.lane.getLength(lane_id)
							# 	distanceFromIntersection = edge_length - cav_lane_position
							cav_wait_time = self.traci.vehicle.getAccumulatedWaitingTime(veh)
							# if cav_wait_time > 2:
							# 	g = 0
							total_waiting_time_cav+=cav_wait_time
							total_cav_speed += self.traci.vehicle.getSpeed(veh)

				
			
				# print("After: CAV Waiting Time --" + str(total_cav_speed))
				rl_wait_time = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
				self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
				# if rl_wait_time > 2:
				# 	g = 0
				# print("Current Waiting Time for RL = ",self._currentRLWaitingTimeForSpecificRLAgent[rl_agent])
				# print("Current Waiting Time for CAV = ",total_waiting_time_cav)
				self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time_cav
				# self._currentOverAllWaitingTime[rl_agent] = total_waiting_time
				agent_lane_pos = self.traci.vehicle.getLanePosition(rl_agent)
				self._AfterCAVSpeed[rl_agent] = total_cav_speed
				edge_id = self.traci.vehicle.getRoadID(rl_agent)
				#check if agent is on a priority lane. Count only those CAV's for it. RL agent on other lanes will have CAV count as zero
				# lane_id = self.traci.vehicle.getLaneID(rl_agent)
				cavCount = 0
				# if lane_id.split("_")[1] == "0":
				all_cav_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
				for cav in all_cav_vehicle:
					priority_type = self.traci.vehicle.getTypeID(cav)
					if priority_type=="cav-priority": 
						cav_lane_position = self.traci.vehicle.getLanePosition(cav)
						diff = agent_lane_pos - cav_lane_position
						if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
							cavCount+=1
				self._numberOfCAVWithinClearingDistanceOnPLAfter[rl_agent] = cavCount
				if lane_id.split("_")[1] == "0":
					self._numberOfCAVWithinClearingDistanceAfter[rl_agent] = cavCount
				else:
					self._numberOfCAVWithinClearingDistanceAfter[rl_agent] = 0
     
	def collectObservationPerStep(self):
		elapsed_simulation_time = self.traci.simulation.getTime()
		allVehicleList = self.traci.vehicle.getIDList()
		# self._collisionCount+= self.traci.simulation.getEmergencyStoppingVehiclesNumber()
		# self._collisionVehicleID.append(self.traci.simulation.getEmergencyStoppingVehiclesIDList())
		self._collisionCount+= self.traci.simulation.getEndingTeleportNumber()

		# self._collisionVehicleID.append(self.traci.simulation.getEmergencyStoppingVehiclesIDList())
		
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle = self.getSplitVehiclesList(allVehicleList)
		
				
		# ld="det_-15_0_1_passenger"
		# temp_list = self.traci.inductionloop.getIntervalVehicleIDs(ld)
		# lane_id = self.traci.vehicle.getLaneID("RL_218")
		# for v in temp_list:
		# 	if v.find("cav")!=-1:
		# 		self._cavFlowCounter+= self.traci.inductionloop.getIntervalVehicleNumber(ld)

		avg_speed_rl=0;currentWaitingTime_rl=0
		total_lane_change_all_RL = 0
		total_lane_change_all = 0
		rl_counter = 0


		for rl_agent in self._rl_vehicleID:
			rledge = self.traci.vehicle.getRoadID(rl_agent)
			if rledge=="-15" or rledge=="-5" or rledge=="-23" or rledge=="-3":
				rl_counter+=1
				avg_speed_rl+=self.traci.vehicle.getSpeed(rl_agent)
				currentWaitingTime_rl += self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
		

		if rl_counter > 0:
			self._avg_speed_rl += avg_speed_rl/rl_counter
			self._currentWaitingTime_rl += currentWaitingTime_rl/rl_counter
		else:
			self._avg_speed_rl += 0
			self._currentWaitingTime_rl += 0

		avg_speed_cav = 0;currentWaitingTime_cav=0
	
		
		cav_counter = 0
		for cav_agent in self._cav_vehicleID:
			cav_edge = self.traci.vehicle.getRoadID(cav_agent)
			if cav_edge=="-15" or cav_edge=="-5" or cav_edge=="-23" or cav_edge=="-3":
				cav_counter+=1
				avg_speed_cav+=self.traci.vehicle.getSpeed(cav_agent)
				currentWaitingTime_cav += self.traci.vehicle.getAccumulatedWaitingTime(cav_agent)
		if cav_counter > 0:
			self._avg_speed_cav += avg_speed_cav/cav_counter
			self._currentWaitingTime_cav += currentWaitingTime_cav/cav_counter
		else:
			self._avg_speed_cav += 0
			self._currentWaitingTime_cav += 0



	def getTestStats(self):
	
		# if  self._episodeStep!=249: # to avoid writing the last line of CSV. 
		avg_delay_RL=0;avg_speed_RL=0;avg_delay_NPC=0;avg_speed_NPC=0;avg_occupancy_priorityLane=0;avg_PMx_emission=0;total_lane_change_number=0
		avg_delay_CAV=0;avg_speed_CAV=0;avg_delay_Heuristic=0;avg_speed_Heuristic=0;avg_delay_ALLButCAV=0;avg_speed_AllButCAV=0
		
		# avg_delay_CAV = self._currentTimeLoss_cav/self.action_steps
		avg_delay_CAV = self._currentWaitingTime_cav/self.action_steps
		avg_speed_CAV = self._avg_speed_cav/300

		# avg_delay_RL = self._currentTimeLoss_rl/self.action_steps
		avg_delay_RL = self._currentWaitingTime_rl/self.action_steps
		avg_speed_RL = self._avg_speed_rl/300

		# print(self._collisionCounter)
		#throughput computation using loop detector
		throughput = 0
		all_LD = self.traci.inductionloop.getIDList()
		LD_counter=0
		for ld in all_LD:			
			if ld=="det_-15_0_1_passenger" or ld=="det_-15_1_1_passenger" or ld=="det_-15_2_1_passenger" or \
				ld=="det_-23_0_1_passenger" or ld=="det_-23_1_1_passenger" or ld=="det_-23_2_1_passenger" or \
				ld=="det_-3_0_1_passenger" or ld=="det_-3_1_1_passenger" or ld=="det_-3_2_1_passenger" or \
				ld=="det_-5_0_1_passenger" or ld=="det_-5_1_1_passenger" or ld=="det_-5_2_1_passenger":
				LD_counter+=1
				throughput += self.traci.inductionloop.getLastIntervalVehicleNumber(ld)		#it was getlastInterval. check	
			# LD_counter+=1
			# throughput += self.traci.inductionloop.getLastIntervalVehicleNumber(ld)
			
		# ld =="det_-3_0_1_passenger" or ld=="det_-3_1_1_passenger" or ld =="det_-3_2_1_passenger" or 
		if LD_counter > 0:
			self._average_throughput = throughput/LD_counter

		avg_throughput = (self._average_throughput*3600)/(300)

	
		headers = ['Avg_WaitingTime_CAV (5 mins)','Avg_WaitingTime_RL (5 mins)','avg_speed_CAV','avg_speed_RL','Throughput','total_lane_change_number_all (5 mins)','total_lane_change_number_RL (5 mins)','total_collision (5 mins)','Episode_Step',"Seed"]
		values = [avg_delay_CAV,avg_delay_RL,avg_speed_CAV,avg_speed_RL,avg_throughput,self._average_LaneChange_number_all,self._average_LaneChange_number_rl,self._collisionCount/300,self._episodeStep,self._sumo_seed]
		
		self._currentTimeLoss_cav=0;self._avg_speed_cav=0;self._currentTimeLoss_rl=0;self._avg_speed_rl=0;self._currentTimeLoss_npc=0;self._avg_speed_npc=0
		self._average_priorityLane_occupancy=0;self._average_throughput=0;self._average_PMx_emission=0;self._currentTimeLoss_Heuristic=0;self._avg_speed_heuristic=0;self._average_LaneChange_number_all=0;self._average_LaneChange_number_rl=0
		self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0;self._average_LaneChange_number=0;self._collisionCounter=0;self._collisionCount=0
		return headers, values
	
	def make_action(self,actions):
		# assumes actions are one-hot encoded
		agent_actions = []
		for i in range(0,self.n): 
			index = np.argmax(actions[i])
			agent_actions.append(index)
		return agent_actions
	
	def step(self,action_n):

		# print("--------Inside STEP-----------")
		obs_n = []
		reward_n = []
		newReward_n = []
		done_n = []
		info_n = {'n':[]}
		actionFlag = True
		for agent in self.agents:
			agent.done = False

		self._episodeStep+=3 # it was 1. Made it to 3 for teststat logs. Might break train
		self._sumo_step = 0
		
		if actionFlag == True:
			temp_action_dict = {}
			simple_actions = self.make_action(action_n)
			# print(simple_actions)
			for i, agent in enumerate(self.agents):
				self.lastActionDict[f'RL_{agent.id}'] = simple_actions[i]
			
			self.collectObservation(True)		#Observation before taking an action - lastTimeStepFlag
			self._set_action()			
			# print(simple_actions)
			actionFlag = False
		
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
		# self.traci.simulation.saveState('sumo_configs/savedstate.xml')
		# self.initializeRLAgentStartValues()
	
		vehicleCount = 0
		
		# if len(self._rlLaneID) == 0:
		self._rlLaneID.clear()
		for rl_agent in self._rl_vehicleID:			
			self._rlLaneID[rl_agent] = self.traci.vehicle.getLaneID(rl_agent)

		self._allVehLaneIDBefore.clear()
		for veh in allVehicleList:			
			self._allVehLaneIDBefore[veh] = self.traci.vehicle.getLaneID(veh)

			
		self._collisionVehicleID.clear()
		while self._sumo_step <= self.action_steps:
			# advance world state
			self.collectObservationPerStep()
			self.traci.simulationStep()
			self._sumo_step +=1	
			# self.collectObservation(False) ##Observation at each step till the end of the action step count (for reward computation) - lastTimeStepFlag lastTimeStepFlag
			# self.keepRLAgentLooping()
			# for loop in self.traci.inductionloop.getIDList():
			# 	vehicleCount +=self.traci.inductionloop.getLastStepVehicleNumber(loop)
			
		# print("Average Number of Vehicles per edge in five minutes are :",vehicleCount/len(self.traci.inductionloop.getIDList()))
		# print(len(self.traci.inductionloop.getIDList()))
		total_lane_change_all = 0
		total_lane_change_rl = 0
		self.collectObservation(False) #lastTimeStepFlag
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
		
		for veh in allVehicleList:
			vehedge = self.traci.vehicle.getRoadID(veh)
			if vehedge=="-15" or vehedge=="-5" or vehedge=="-23" or vehedge=="-3":
				if veh in self._allVehLaneIDBefore:
					if self._allVehLaneIDBefore[veh] != self.traci.vehicle.getLaneID(veh):
						total_lane_change_all+=1

		self._average_LaneChange_number_all+= total_lane_change_all

		for rl_agent in self._rl_vehicleID:			
			vehedge = self.traci.vehicle.getRoadID(rl_agent)
			if rl_agent in self._rlLaneID:
				if vehedge=="-15" or vehedge=="-5" or vehedge=="-23" or vehedge=="-3":
					if self._rlLaneID[rl_agent] != self.traci.vehicle.getLaneID(rl_agent):
						total_lane_change_rl+=1

		self._average_LaneChange_number_rl+= total_lane_change_rl
		# print("Total npc: " + str(len(self._npc_vehicleID)) + "Total RL agent: " + str(len(self._rl_vehicleID)))
		
		# if len(self._rl_vehicleID)!=self.n:
		# 	print("Total RL agent before loadState: " + str(len(self._rl_vehicleID)))
		# 	self.traci.simulation.loadState('sumo_configs/savedstate.xml')
		# 	print("Total RL agent After loadState: " + str(len(self._rl_vehicleID)))
			# self.keepRLAgentLooping()


			
		# allVehicleList = self.traci.vehicle.getIDList()
		# self._npc_vehicleID,self._rl_vehicleID = self.getSplitVehiclesList(allVehicleList)
		# print("Total npc: " + str(len(self._npc_vehicleID)) + "Total RL agent: " + str(len(self._rl_vehicleID)))
  
  
		# if self._isTestFlag==False:
		# 	if ratioOfHaltVehicle >0.90:
		# 		print("Reset the Episode")
		# 		for agent in self.agents:
		# 			agent.done= True
       
		for agent in self.agents:
			obs_n.append(self._get_obs(agent))	
			# print(self._get_obs(agent))		
			reward_n.append(self._get_reward(agent))
			# print(self._get_reward(agent))
			done_n.append(self._get_done(agent))

			info_n['n'].append(self._get_info(agent))

		if self._reward_type == "Global":
			self._currentReward = reward_n
			reward = np.sum(reward_n)/self.n
			newReward_n = [reward] *self.n
		elif self._reward_type == "Individual":
			self._currentReward = reward_n
			newReward_n = reward_n
		else:
			#find sum of all local agent reward including the one in concern
			for agent in self.agents:
				agentReward = reward_n[agent.id]
				agentList = self._listOfLocalRLAgents[agent.name]
				reward = 0
				for ag in agentList:
					id = int(ag.split("_")[1])
					reward +=reward_n[id]
				
				if len(agentList)>0:
					finalReward = (agentReward+reward)/(len(agentList)+1)
				else:
					finalReward = agentReward
				newReward_n.append(finalReward)
   
		# print("Reward = " + str(reward_n))
		# self._lastReward = reward_n[0]
		# # print("reward: " + str(self._lastReward))
		# print(newReward_n)
		return obs_n, newReward_n, done_n, info_n

	# set env action for a particular agent
	def _set_action(self,time=None):
		# process action
		#index 0 = # set default
		#index 1 = # set priority
  
		if self.SotaFlag==True:
			self.setRLAgentTogglePriority() # to simulate human decision-making
		for agent in self.agents: #loop through all agent
			agent_id = f'RL_{agent.id}'
			action = self.lastActionDict[agent_id]
			# print("action: " + str(action))
			# if self.SotaFlag==True:
			# action = 2
			# if action==2:
			# print(action)
		
			if action == 0:
				if self.traci.vehicle.getTypeID(agent_id)!="rl-default":
					self._collisionCounter+=1
					self.traci.vehicle.setType(agent_id,"rl-default")
				lane_index = self.traci.vehicle.getLaneIndex(agent_id)
				if lane_index==0:
					# before = self.traci.vehicle.getLaneChangeStatePretty(agent_id,1)
					self.traci.vehicle.changeLane(agent_id,1,0)
					# after = self.traci.vehicle.getLaneChangeStatePretty(agent_id,1)
					# result = self.traci.vehicle.couldChangeLane(agent_id,1)
					# lane_index_after = self.traci.vehicle.getLaneIndex(agent_id)
					# p = 0
					
				
			elif action == 1:
				# lane_index = self.traci.vehicle.getLaneIndex(agent_id)
				# if lane_index!=0 and self.traci.vehicle.getTypeID(agent_id)!="rl-priority":
				if self.traci.vehicle.getTypeID(agent_id)!="rl-priority":
					self._collisionCounter+=1
					self.traci.vehicle.setType(agent_id,"rl-priority")
				lane_index = self.traci.vehicle.getLaneIndex(agent_id)
				if lane_index!=0:
					self.traci.vehicle.changeLane(agent_id,0,0)
			# if self.traci.vehicle.getTypeID(agent_id)=="rl-priority": #check if agent  
			# 	if action == 0:
			# 		self.traci.vehicle.setType(agent_id,"rl-default")
			# 		# self.traci.vehicle.rerouteEffort(agent_id)
			# 		# self.traci.vehicle.updateBestLanes(agent_id)
			# 		# if self.edgeIdInternal(self.traci.vehicle.getLaneID(agent_id)) == False:
			# 		# 	bestLanes = self.traci.vehicle.getBestLanes(agent_id)
			# 		# 	self.traci.vehicle.changeLane(agent_id,1,self._laneChangeAttemptDuration) 
			# 			# if bestLanes[0][1] > bestLanes[1][1]: # it checks the length that can be driven without lane
			# 			# 	#change for the prospective lanes (measured from the start of that lane). Higher value is preferred. 
			# 			# 	self.traci.vehicle.changeLane(agent_id,0,self._laneChangeAttemptDuration) 
			# 			# else:
			# 			# 	self.traci.vehicle.changeLane(agent_id,1, self._laneChangeAttemptDuration)
			# 		# print("Priority Removed")
			# 	elif action == 1:
			# 		pass # do nothing
			# else:
			# 	if action == 0:
			# 		self.traci.vehicle.setType(agent_id,"rl-priority")
			# 		# self.traci.vehicle.rerouteEffort(agent_id)
			# 		# self.traci.vehicle.updateBestLanes(agent_id)
			# 		# self.traci.vehicle.changeLane(agent_id,0,self._laneChangeAttemptDuration) 
			# 	elif action == 1:
			# 		pass # do nothing
					
	
	def initSimulator(self,withGUI,portnum):
		if withGUI:
			import traci
		else:
			try:
				import libsumo as traci
			except:
				import traci
		seed = self._sumo_seed
  

		if self._isTestFlag==True:
			self.sumoCMD = ["-c", self.sumoConfig, "-r", self._routeFileName, "--waiting-time-memory",str(self.action_steps+1),"--time-to-teleport", str(1),"--scale",str(1),
			"-W","--lanechange-output",f"laneChange_stats_{seed}.xml","--collision.action","teleport","--statistic-output","output.xml","--edgedata-output",f"edge_stats_{seed}.xml"]
		else:				
			self.sumoCMD = ["-c", self.sumoConfig, "-r", self._routeFileName, "--waiting-time-memory",str(self.action_steps+1),"--time-to-teleport", str(-1),"--scale",str(1),
				"-W","--collision.action","none"]
		# "-W", "--default.carfollowmodel", "IDM",
		if withGUI:
			sumoBinary = checkBinary('sumo-gui')
			self.sumoCMD += ["--start"]
			# self.sumoCMD += ["--start", "--quit-on-end"]
		else:	
			sumoBinary = checkBinary('sumo')

		# print(sumoBinary)
		# sumoConfig = "sumo_configs/sim.sumocfg"
# "--lanechange.duration",str(2),

		random.seed(seed)
		traci.start([sumoBinary] + ["--seed", str(seed)] + self.sumoCMD)
		return traci

	def close(self):
		print("RL_Counter :" + str(self._rl_counter))
		print("CAV Flow " + str(self._cavFlowCounter))
		self.traci.close()
		sys.stdout.flush()

