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
sys.path.append('../') #allows loading of agent.py
# from agent import Agent
import xml.etree.ElementTree as ET
import math
from itertools import combinations
import sumolib
class Agent:
    def __init__(self, env, n_agent, edge_agent=None):
        """Dummy agent object"""
        self.edge_agent = edge_agent
        self.traci = env.traci
        self.env = env

        self.id = n_agent
        self.name = f'RL_{self.id}'

class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	
	def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',testStatAccumulation=10,testFlag='False',simulation_end=36000):
		self.pid = os.getpid()
		self.sumoCMD = []
		self._simulation_end = simulation_end
		self._mode = mode
		self._testStatAccumulation = testStatAccumulation
		if testFlag == False:
			self._networkFileName = "sumo_configs/LargeTestNetwork.net.xml"
			self._routeFileName = "sumo_configs/LargeTestNetwork.rou.xml"   
		else:
			self._networkFileName = "sumo_configs/LargeTestNetwork.net.xml"
			self._routeFileName = "sumo_configs/LTN_Density1_CAV20.rou.xml"   

		self._episodeStep = 0
		self._isTestFlag = testFlag
		# self._seed(40)
		# np.random.seed(42)
		self._sumo_seed = 42
		self._reward_type = "Global" 
		# self._reward_type = "Local" 
		self.withGUI = mode
		self.action_steps = 30	
		self._warmup_steps = 1199
		self.traci = self.initSimulator(self.withGUI, self.pid)
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
		self._net = sumolib.net.readNet(self._networkFileName,withInternal=True)
		# set required vectorized gym env property
		self.n = 50 #read it from the route file
		self.lastActionDict = {}
		self.lastTimeLossRLAgents = {}
		self._lastOverAllTimeLoss = {}
		self._currentOverAllTimeLoss = {}
		self._lastOverAllWaitingTime = {}
		self._lastCAVWaitingTimeForSpecificRLAgent = {}
		self._currentCAVWaitingTimeForSpecificRLAgent = {}
		self._currentRLWaitingTimeForSpecificRLAgent = {}
		self._lastRLWaitingTimeForSpecificRLAgent = {}
		self._currentOverAllWaitingTime = {}
		self._listOfVehicleIdsInConcern = {}
		self._numberOfCAVWithinClearingDistance = {}
		self._numberOfCAVApproachingIntersection = {}
		self._beforePriorityForRLAgent = {}
		self._afterPriorityForRLAgent = {}
		self._listOfLocalRLAgents = {}
		self._releventEdgeId = []
		self._timeLossThreshold = 60
		self._lane_clearing_distance_threshold = 30
		self._lane_clearing_distance_threshold_RL = 5
		self._lane_clearing_distance_threshold_state = 100
		self._laneChangeAttemptDuration = 25 #seconds
		self._weightCAVPriority = 1 #3
		self._weightRLWeightingTime = 1
		self._weightCAVWeightingTime = 1 #2
		self._average_throughput = 0
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
		self._average_PMx_emission = 0

		self._allEdgeIds = self.traci.edge.getIDList()
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
		# self._num_observation = [6,6]
		self._num_observation = 10
		self._num_actions = 2
		# self._num_actions = [len(priority_actions), len(priority_actions)]
		# self._num_observation = [len(Agent(self, i, self.edge_agents[0]).getState()) for i in range(self._num_lane_agents)]*len(self.edge_agents)
		self.action_space = []
		self.observation_space = []
		for i in range(self.n):
			self.action_space.append(spaces.Discrete(self._num_actions)) #action space			
			self.observation_space.append(spaces.Box(low=0, high=1, shape=(self._num_observation,)))# observation space
			
		self.agents = self.createNAgents()

		# parse the net

	def createNAgents(self):
		agents = [Agent(self, i) for i in range(self.n)]

		return agents
	
	def getState(self,agent_id):
		"""
		Retrieve the state of the network from sumo. 
		"""
		state = []
		# State = { Number of vehicle with priority lane access, number of vehicle without priority lane access, It’s own priority lane access, Avg. delay over all edges, number of emergency vehicle, number of public buses)
		#Get the edgeID on which the RL agent is:
		# agent_id = "RL_0"
		# print(agent_id)
		#Get the edgeID on which the RL agent is:
		edge_id = self.traci.vehicle.getRoadID(agent_id)
		lane_id = self.traci.vehicle.getLaneID(agent_id)			
		priorityLane_id = edge_id + "_0"
		# print(edge_id,agent_id)
		nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
		# now found the edges that is incoming to this junction
		incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()

		#count all  CAV vehicle behind this RL agent and within clearingThresholdDistance
		agent_lane_pos = self.traci.vehicle.getLanePosition(agent_id)
		all_vehicle = self.traci.edge.getLastStepVehicleIDs(edge_id)
		cavCount = 0
		for cav in all_vehicle:
			priority_type = self.traci.vehicle.getTypeID(cav)
			if priority_type=="cav-priority":
				cav_lane_position = self.traci.vehicle.getLanePosition(cav)
				diff = agent_lane_pos - cav_lane_position
				if diff<= self._lane_clearing_distance_threshold_state and diff > 0:
					cavCount+=1

		self._numberOfCAVWithinClearingDistance[agent_id] = cavCount

		#find next edge in the route for the RL agent
		list = self.traci.vehicle.getNextLinks(agent_id)
		nextLane = list[0][0]
		nextPriorityLane_id = nextLane.split("_")[0] + "_0"
		#check occupancy of that lane
		occupancyNextLane = self.traci.lane.getLastStepOccupancy(nextLane)
		occupancyCurrentLane = self.traci.lane.getLastStepOccupancy(lane_id)
		occupancyCurrentPriorityLane = self.traci.lane.getLastStepOccupancy(priorityLane_id)
		occupancyNextPriorityLane = self.traci.lane.getLastStepOccupancy(nextPriorityLane_id)

		# # print(edge_id)
		# #Get the intersection the RL agent is going towards:
		# # retrieve the successor edges of an edge
		# nextEdges = self._net.getEdge(edge_id).getOutgoing()
		# edge_list = [e.getID() for e in nextEdges] # list of all edges excluding internal
		# nextIncomingEdges = self._net.getEdge(edge_id).getIncoming()
		# edge_list_incoming = [e.getID() for e in nextIncomingEdges] # list of all edges excluding internal
		priorityVehicleCount = 0
		localRLVehicleCount = 0
		nonPriorityVehicleCount = 0
		# total_waiting_time = 0
		accumulated_time_loss = 0
		localRLAgentList = []
		normalization_totalNumberOfVehicle = 100
		normalization_totalNumberOfCAV = 50
		elapsed_simulation_time = self.traci.simulation.getTime()
		edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
		all_cav_count = 0
		for e_id in edge_list_incoming:   
			all_vehicle = self.traci.edge.getLastStepVehicleIDs(e_id)
			for veh in all_vehicle:
				elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)			
				accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh) / (elapsed_simulation_time - elapsed_vehicle_time)
				# total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
				priority_type = self.traci.vehicle.getTypeID(veh)
				if priority_type=="passenger-priority" or priority_type=="rl-priority" or priority_type=="cav-priority" or priority_type=="heuristic-priority":
					priorityVehicleCount+=1
					if priority_type=="rl-priority":
						localRLVehicleCount+=1
						localRLAgentList.append(veh)
					if priority_type=="cav-priority":
						cav_lane_position = self.traci.vehicle.getLanePosition(veh)
						diff = agent_lane_pos - cav_lane_position
						if diff <=self._lane_clearing_distance_threshold_state and diff>0:
							all_cav_count+=1
				else:
					nonPriorityVehicleCount+=1
		self._listOfLocalRLAgents[agent_id] = localRLAgentList	
		self._numberOfCAVApproachingIntersection[agent_id] = all_cav_count
		elapsed_its_own_time = self.traci.vehicle.getDeparture(agent_id)	
		itsOwnTImeLoss = self.traci.vehicle.getTimeLoss(agent_id) / (elapsed_simulation_time - elapsed_its_own_time)
		# self.lastTimeLoss[agent_id] = itsOwnTImeLoss
		if self.traci.vehicle.getTypeID(agent_id)=="rl-priority":
			itsPriorityAccess = 1			
		else:
			itsPriorityAccess = 0
		# print(self._sumo_step)
		rlObs=0;cavObs=0
		if len(self._currentRLWaitingTimeForSpecificRLAgent)>0:
			if self._currentRLWaitingTimeForSpecificRLAgent[agent_id] - self._lastRLWaitingTimeForSpecificRLAgent[agent_id] > 0:
				rlObs = 0
			else:
				rlObs = 1
	
			if self._currentCAVWaitingTimeForSpecificRLAgent[agent_id] - self._lastCAVWaitingTimeForSpecificRLAgent[agent_id]>0:
				cavObs = 0
			else:
				cavObs = 1
		lane_index = int(lane_id.split("_")[1])
		# state = [itsOwnTImeLoss/self.action_steps,itsPriorityAccess,priorityVehicleCount/normalization_totalNumberOfVehicle,nonPriorityVehicleCount/normalization_totalNumberOfVehicle,accumulated_time_loss/self.action_steps,cavCount/normalization_totalNumberOfCAV,all_cav_count/normalization_totalNumberOfVehicle]
		state = [itsPriorityAccess,
		   lane_index/2,
		   occupancyCurrentLane,
		   occupancyCurrentPriorityLane,
		   occupancyNextLane,
		   occupancyNextPriorityLane,
		   localRLVehicleCount/self.n,
		   nonPriorityVehicleCount/normalization_totalNumberOfVehicle,
		   cavCount/normalization_totalNumberOfCAV,
		   all_cav_count/normalization_totalNumberOfVehicle
           ]
		# state = [itsPriorityAccess,
		#    localRLVehicleCount/self.n,
		#    nonPriorityVehicleCount/normalization_totalNumberOfVehicle,
		#    cavCount/normalization_totalNumberOfCAV,
		#    all_cav_count/normalization_totalNumberOfVehicle,
     	#    rlObs,
        #    cavObs]
           
          
		# if agent_id == "RL_1":
		# 	print(state)
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
		self._average_PMx_emission = 0
		self._episodeStep = 0	
		self._average_throughput = 0

		# self._npc_vehicleID=0
		# self._rl_vehicleID=0

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
					if diff>= self._lane_clearing_distance_threshold:					
						continue
					else:
						#change priority of heuristic agent as it is inside clearing distance
						# speed = self.traci.vehicle.getSpeed(heuristic)
						if diff > 0:
							# print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
							self.traci.vehicle.setType(heuristic,"heuristic-default")
							flag = True
							bestLanes = self.traci.vehicle.getBestLanes(heuristic)
							counterDefault+=1
							# if bestLanes[0][1] > bestLanes[1][1]: # it checks the length that can be driven without lane
							# 	#change for the prospective lanes (measured from the start of that lane). Higher value is preferred. 
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
			if self.edgeIdInternal(which_lane)==False:
				lane_index = which_lane.split("_")[1]
				which_edge = which_lane.split("_")[0]
				priority_lane = which_edge + str("_0") # find priority lane for that vehicle
				vehicle_on_priority_lane = self.traci.lane.getLastStepVehicleIDs(priority_lane)
				npc_vehicleID,rl_vehicleID, heuristic_vehicleID,cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(vehicle_on_priority_lane)
				heuristic_lane_position = self.traci.vehicle.getLanePosition(rl)
				
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
						if diff > 0:
							# print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
							self.traci.vehicle.setType(rl,"rl-default")
							flag = True
							# bestLanes = self.traci.vehicle.getBestLanes(rl)
							# counterDefault+=1
							# if bestLanes[0][1] > bestLanes[1][1]: # it checks the length that can be driven without lane
							# 	#change for the prospective lanes (measured from the start of that lane). Higher value is preferred. 
							# 	self.traci.vehicle.changeLane(heuristic,0,self._laneChangeAttemptDuration) 
							# else:
							# 	self.traci.vehicle.changeLane(heuristic,1, self._laneChangeAttemptDuration)
							break
				if flag==False: 
					if lane_index!=0 and self.traci.vehicle.getTypeID(rl)!="rl-priority":
						# speed = self.traci.vehicle.getSpeed(heuristic)
						# if speed>0.2:
						self.traci.vehicle.setType(rl,"rl-priority")
						# counterPriority+=1
						# if cav_lane_position ==-999:
						# 	print(str(diff),"--",str(heuristic_lane_position),"--","No CAV present")
						# else:
						# 	print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
						# self.traci.vehicle.changeLane(heuristic,2,50) 
					

		# print("PtoD change =",counterDefault,"  DtoP changes =",counterPriority)


	def reset(self,scenario):		
		print("--------Inside RESET---------")
		self._sumo_step = 0
		self._scenario = scenario
		self.resetAllVariables()
		obs_n = []
		seed = self._sumo_seed
		self.sumoCMD = ["--seed", str(seed),"--waiting-time-memory",str(self.action_steps),"--time-to-teleport", str(-1), "-W", "true",
				"--statistic-output","output.xml"]
		self.sumoCMD += ["--start"]
		self.traci.load(self.sumoCMD + ['-n', self._networkFileName, '-r', self._routeFileName])
		#WARMUP PERIOD
		while self._sumo_step <= self._warmup_steps:
			self.traci.simulationStep() 		# Take a simulation step to initialize	
			# print(self.traci.vehicle.getTimeLoss("RL_9"))
			# if self._sumo_step == 10:
			# 	allVehicleList = self.traci.vehicle.getIDList()
			# 	self._npc_vehicleID,self.original_rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
				
				# self.initializeRLAgentStartValues()
			# 	self.keepRLAgentLooping()
			if self._sumo_step%self.action_steps==0:
				# self.setHeuristicAgentTogglePriority()
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

		if self._numberOfCAVWithinClearingDistance[rl_agent] > 0: #or self._numberOfCAVApproachingIntersection[rl_agent]>0:
			if before_priority=="rl-priority" and after_priority=="rl-default":
				reward = +0.5
			elif before_priority=="rl-priority" and after_priority=="rl-priority":
				reward = -0.5
			elif before_priority=="rl-default" and after_priority=="rl-default":
				reward = +0.5
			elif before_priority=="rl-default" and after_priority=="rl-priority":
				reward = -0.5
		elif self._numberOfCAVWithinClearingDistance[rl_agent] ==0:# and self._numberOfCAVApproachingIntersection[rl_agent]==0:
			if before_priority=="rl-priority" and after_priority=="rl-default":
				reward = -0.5
			elif before_priority=="rl-priority" and after_priority=="rl-priority":
				reward = +0.5
			elif before_priority=="rl-default" and after_priority=="rl-default":
				reward = -0.5
			elif before_priority=="rl-default" and after_priority=="rl-priority":
				reward = +0.5
		
		return reward
	
	def computeCAVAccumulatedWaitingTime(self,rl_agent):
		if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] >0 and self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]> 0:			
			if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]> 0:
			# cav_delay = -(self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]
				cav_delay = -(self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]   
			else:
				if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent]!=self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]:
					cav_delay = +(self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent] - self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent])/self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent]
				else:
					cav_delay=0
		else:
			if self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] == 0: 
				cav_delay = +0.5
			else:
				cav_delay = -0.5

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


	# get reward for a particular agent
	def _get_reward(self,agent):
		agent_id = f'RL_{agent.id}'
		overall_reward = 0
		if len(self.lastActionDict) !=0:				
			# reward_cooperative = self.computeCooperativeReward(agent_id)
			# reward_overallNetwork = self.computeOverallNetworkReward(agent_id)
			reward_cavWaitingTime = self.computeCAVAccumulatedWaitingTime(agent_id)
			reward_RLWaitingTime = self.computeRLAccumulatedWaitingTime(agent_id)
			reward_cav_priority = self.computeCAVReward(agent_id)
			# overall_reward = reward_cooperative + reward_overallNetwork + reward_cav_priority
			# overall_reward = reward_cav_priority
			# print(overall_reward)		
			overall_reward = self._weightCAVPriority*reward_cav_priority + self._weightRLWeightingTime*reward_RLWaitingTime + self._weightCAVWeightingTime*reward_cavWaitingTime
			# overall_reward = reward_cav_priority


		return overall_reward
		
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _get_done(self, agent):  
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

	def collectObservation(self,lastTimeStepFlag):
		#This function collects sum of time loss for all vehicles related to a in-concern RL agent. 
		
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
		elapsed_simulation_time = self.traci.simulation.getTime()
		if lastTimeStepFlag:
			self._listOfVehicleIdsInConcern.clear()
			for rl_agent in self._rl_vehicleID:
				elapsed_its_own_time = self.traci.vehicle.getDeparture(rl_agent)
				itsOwnTImeLoss = self.traci.vehicle.getTimeLoss(rl_agent) / (elapsed_simulation_time - elapsed_its_own_time)
				self.lastTimeLossRLAgents[rl_agent] = itsOwnTImeLoss
				edge_id = self.traci.vehicle.getRoadID(rl_agent)
				self._beforePriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
				accumulated_time_loss = 0
				total_waiting_time=0
				total_waiting_time_cav = 0
				# print(edge_id,agent_id)
				#check if edge_id is internal
				# if self.edgeIdInternal(edge_id):
				# 	print("Internal Edge ID - ",edge_id) #change it to the main edge_id
				vehicle_list =[]
				nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
				# now found the edges that is incoming to this junction
				incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()
				edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
				for e_id in edge_list_incoming:   
					all_vehicle = self.traci.edge.getLastStepVehicleIDs(e_id)
					if len(all_vehicle)>0:
						vehicle_list+=all_vehicle
					for veh in all_vehicle:
						elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)
						accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh)/(elapsed_simulation_time - elapsed_vehicle_time)
						total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
						priority_type = self.traci.vehicle.getTypeID(veh)
						if priority_type=="cav-priority": 
							total_waiting_time_cav+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
				
				self._listOfVehicleIdsInConcern[rl_agent] = vehicle_list
				self._lastRLWaitingTimeForSpecificRLAgent[rl_agent] = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
				# print("Before Waiting Time for RL = ",self._lastRLWaitingTimeForSpecificRLAgent[rl_agent])
				# print("Before Waiting Time for CAV = ",total_waiting_time_cav)
				
				self._lastCAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time_cav				
				self._lastOverAllTimeLoss[rl_agent] = accumulated_time_loss
				self._lastOverAllWaitingTime[rl_agent] = total_waiting_time
		else:
			for rl_agent in self._rl_vehicleID:
				accumulated_time_loss = 0
				total_waiting_time=0
				total_waiting_time_cav=0
				self._afterPriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
				for veh in self._listOfVehicleIdsInConcern[rl_agent]:
					if veh in allVehicleList:
						elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)
						accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh)/(elapsed_simulation_time - elapsed_vehicle_time)
						total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(veh)
						priority_type = self.traci.vehicle.getTypeID(veh)
						if priority_type=="cav-priority": 
							total_waiting_time_cav+=self.traci.vehicle.getAccumulatedWaitingTime(veh)

				self._currentRLWaitingTimeForSpecificRLAgent[rl_agent] = self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
				# print("Current Waiting Time for RL = ",self._currentRLWaitingTimeForSpecificRLAgent[rl_agent])
				# print("Current Waiting Time for CAV = ",total_waiting_time_cav)
				self._currentCAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time_cav
				self._currentOverAllWaitingTime[rl_agent] = total_waiting_time

	def collectObservationPerStep(self):
		elapsed_simulation_time = self.traci.simulation.getTime()
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle = self.getSplitVehiclesList(allVehicleList)
		
		avg_speed_heuristic=0;currentTimeLoss_Heuristic=0;currentWaitingTime_Heuristic=0
		for heuristic_agent in self._heuristic_vehicleID:
			avg_speed_heuristic+=self.traci.vehicle.getSpeed(heuristic_agent)
			elapsed_its_own_time = self.traci.vehicle.getDeparture(heuristic_agent)
			currentTimeLoss_Heuristic += self.traci.vehicle.getTimeLoss(heuristic_agent)
			currentWaitingTime_Heuristic += self.traci.vehicle.getWaitingTime(heuristic_agent)
		self._currentTimeLoss_Heuristic += currentTimeLoss_Heuristic/len(self._heuristic_vehicleID)
		self._avg_speed_heuristic += avg_speed_heuristic/len(self._heuristic_vehicleID)
		self._currentWaitingTime_Heuristic += currentWaitingTime_Heuristic/len(self._heuristic_vehicleID)

		avg_speed_rl=0;currentTimeLoss_rl=0;currentWaitingTime_rl=0
		for rl_agent in self._rl_vehicleID:
			avg_speed_rl+=self.traci.vehicle.getSpeed(rl_agent)
			elapsed_its_own_time = self.traci.vehicle.getDeparture(rl_agent)
			currentTimeLoss_rl += self.traci.vehicle.getTimeLoss(rl_agent)
			currentWaitingTime_rl += self.traci.vehicle.getWaitingTime(rl_agent)
		self._currentTimeLoss_rl += currentTimeLoss_rl/len(self._rl_vehicleID)
		self._avg_speed_rl += avg_speed_rl/len(self._rl_vehicleID)
		self._currentWaitingTime_rl += currentWaitingTime_rl/len(self._rl_vehicleID)

		avg_speed_npc=0;currentTimeLoss_npc=0;currentWaitingTime_npc=0
		for npc_agent in self._npc_vehicleID:
			avg_speed_npc+=self.traci.vehicle.getSpeed(npc_agent)
			elapsed_its_own_time = self.traci.vehicle.getDeparture(npc_agent)
			currentTimeLoss_npc += self.traci.vehicle.getTimeLoss(npc_agent)
			currentWaitingTime_npc += self.traci.vehicle.getWaitingTime(npc_agent)
		self._currentTimeLoss_npc += currentTimeLoss_npc/len(self._npc_vehicleID)
		self._avg_speed_npc += avg_speed_npc/len(self._npc_vehicleID)
		self._currentWaitingTime_npc += currentWaitingTime_npc/len(self._npc_vehicleID)

		avg_speed_cav = 0;currentTimeLoss_cav=0;currentWaitingTime_cav=0
		for cav_agent in self._cav_vehicleID:
			avg_speed_cav+=self.traci.vehicle.getSpeed(cav_agent)
			elapsed_its_own_time = self.traci.vehicle.getDeparture(cav_agent)
			currentTimeLoss_cav += self.traci.vehicle.getTimeLoss(cav_agent) 
			currentWaitingTime_cav += self.traci.vehicle.getWaitingTime(cav_agent)
		self._currentTimeLoss_cav += currentTimeLoss_cav/len(self._cav_vehicleID)
		self._avg_speed_cav += avg_speed_cav/len(self._cav_vehicleID)
		self._currentWaitingTime_cav += currentWaitingTime_cav/len(self._cav_vehicleID)


		average_priorityLane_occupancy = 0;average_PMx_emission=0
		for edge in self._releventEdgeId:
			#check only for priority lane
			priority_lane = edge + "_0"
			average_priorityLane_occupancy += self.traci.lane.getLastStepOccupancy(priority_lane)
			average_PMx_emission += self.traci.edge.getPMxEmission(edge)
		self._average_priorityLane_occupancy += average_priorityLane_occupancy/len(self._releventEdgeId)
		self._average_PMx_emission += average_PMx_emission/len(self._releventEdgeId)

		# for rl_agent in self._rl_vehicleID:
		# 	total_waiting_time = 0
		# 	edge_id = self.traci.vehicle.getRoadID(rl_agent)
		# 	nextNodeID = self._net.getEdge(edge_id).getToNode().getID() # gives the intersection/junction ID
		# 	# now found the edges that is incoming to this junction
		# 	incomingEdgeList = self._net.getNode(nextNodeID).getIncoming()
		# 	edge_list_incoming = [e.getID() for e in incomingEdgeList] # list of all edges excluding internal
		# 	for e_id in edge_list_incoming:   
		# 		all_vehicle = self.traci.edge.getLastStepVehicleIDs(e_id)
		# 		for cav in all_vehicle:
		# 			priority_type = self.traci.vehicle.getTypeID(cav)
		# 			if priority_type=="cav-priority": 
		# 				total_waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(cav)

		# 	self._CAVWaitingTimeForSpecificRLAgent[rl_agent] = total_waiting_time


	def getTestStats(self):
	
			avg_delay_RL=0;avg_speed_RL=0;avg_delay_NPC=0;avg_speed_NPC=0;avg_occupancy_priorityLane=0;avg_PMx_emission=0
			avg_delay_CAV=0;avg_speed_CAV=0;avg_delay_Heuristic=0;avg_speed_Heuristic=0;avg_delay_ALLButCAV=0;avg_speed_AllButCAV=0
			
			# avg_delay_CAV = self._currentTimeLoss_cav/self.action_steps
			avg_delay_CAV = self._currentWaitingTime_cav/self.action_steps
			avg_speed_CAV = self._avg_speed_cav/self.action_steps

			# avg_delay_RL = self._currentTimeLoss_rl/self.action_steps
			avg_delay_RL = self._currentWaitingTime_rl/self.action_steps
			avg_speed_RL = self._avg_speed_rl/self.action_steps

			# avg_delay_NPC = self._currentTimeLoss_npc/self.action_steps
			avg_delay_NPC = self._currentWaitingTime_npc/self.action_steps
			avg_speed_NPC = self._avg_speed_npc/self.action_steps

			avg_occupancy_priorityLane = self._average_priorityLane_occupancy/self.action_steps
			avg_PMx_emission = self._average_PMx_emission/self.action_steps
   
			#throughput computation using loop detector
			throughput = 0
			all_LD = self.traci.inductionloop.getIDList()
			for ld in all_LD:
				throughput += self.traci.inductionloop.getLastIntervalVehicleNumber(ld)
			if len(all_LD) > 0:
				self._average_throughput = throughput/len(all_LD)

			avg_throughput = (self._average_throughput*3600)/(self.action_steps*self._testStatAccumulation)
   
			avg_throughput = self._average_throughput/self.action_steps

			# avg_delay_Heuristic = self._currentTimeLoss_Heuristic/self.action_steps
			avg_delay_Heuristic = self._currentWaitingTime_Heuristic/self.action_steps
			avg_speed_Heuristic = self._avg_speed_heuristic/self.action_steps

			avg_delay_ALLButCAV = (avg_delay_RL + avg_delay_NPC + avg_delay_Heuristic)/3
			avg_speed_AllButCAV = (avg_speed_RL + avg_speed_NPC + avg_speed_Heuristic)/3
			
			# headers = ['avg_delay_RL', 'avg_speed_RL','avg_delay_NPC', 'avg_speed_NPC','congestion(avg_occupancy_network))','avg_PMx_emission']
			# values = [avg_delay_RL, avg_speed_RL, avg_delay_NPC,avg_speed_NPC,avg_occupancy_network,avg_PMx_emission]
			headers = ['avg_delay_CAV','avg_delay_RL','avg_speed_CAV','avg_speed_RL','Throughput','avg_PMx_emission','Episode_Step']
			values = [avg_delay_CAV,avg_delay_RL,avg_speed_CAV,avg_speed_RL,avg_throughput,avg_PMx_emission,self._episodeStep-1]
			
			self._currentTimeLoss_cav=0;self._avg_speed_cav=0;self._currentTimeLoss_rl=0;self._avg_speed_rl=0;self._currentTimeLoss_npc=0;self._avg_speed_npc=0
			self._average_priorityLane_occupancy=0;self._average_throughput=0;self._average_PMx_emission=0;self._currentTimeLoss_Heuristic=0;self._avg_speed_heuristic=0
			self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0
			return headers, values
	
	def make_action(self,actions):
		agent_actions = []
		for i in range(0,self.n): 
			index = np.argmax(actions[i])
			agent_actions.append(index)
		return agent_actions
	
	def _step(self,action_n):

		print("--------Inside STEP-----------")
		obs_n = []
		reward_n = []
		newReward_n = []
		done_n = []
		info_n = {'n':[]}
		actionFlag = True
		for agent in self.agents:
			agent.done = False

		self._episodeStep+=1
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
		
		
		self.traci.simulation.saveState('sumo_configs/savedstate.xml')
		# self.initializeRLAgentStartValues()
		vehicleCount = 0
		while self._sumo_step <= self.action_steps:
			# advance world state
			# self.collectObservationPerStep()
			self.traci.simulationStep()
			self._sumo_step +=1	
			# self.collectObservation(False) ##Observation at each step till the end of the action step count (for reward computation) - lastTimeStepFlag lastTimeStepFlag
			# self.keepRLAgentLooping()
			# for loop in self.traci.inductionloop.getIDList():
			# 	vehicleCount +=self.traci.inductionloop.getLastStepVehicleNumber(loop)
			
		# print("Average Number of Vehicles per edge in five minutes are :",vehicleCount/len(self.traci.inductionloop.getIDList()))
		# print(len(self.traci.inductionloop.getIDList()))

		self.collectObservation(False) #lastTimeStepFlag
		allVehicleList = self.traci.vehicle.getIDList()
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
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
					finalReward = (reward)/len(agentList)
				else:
					finalReward = agentReward
				newReward_n.append(finalReward)
   
		# print("Reward = " + str(reward_n))
		# self._lastReward = reward_n[0]
		# # print("reward: " + str(self._lastReward))
		# print(reward_n)
		return obs_n, newReward_n, done_n, info_n

	# set env action for a particular agent
	def _set_action(self,time=None):
		# process action
		#index 0 = # Toggle Priority
		#index 1 = # do nothing
		# self.setHeuristicAgentTogglePriority() # to simulate human decision-making
		for agent in self.agents: #loop through all agent
			agent_id = f'RL_{agent.id}'
			action = self.lastActionDict[agent_id]
			# action = 1
			# if action==2:
			# print(action)
			if self.traci.vehicle.getTypeID(agent_id)=="rl-priority": #check if agent  
				if action == 0:
					self.traci.vehicle.setType(agent_id,"rl-default")
					
					# if self.edgeIdInternal(self.traci.vehicle.getLaneID(agent_id)) == False:
					# 	bestLanes = self.traci.vehicle.getBestLanes(agent_id)
					# 	self.traci.vehicle.changeLane(agent_id,1,self._laneChangeAttemptDuration) 
						# if bestLanes[0][1] > bestLanes[1][1]: # it checks the length that can be driven without lane
						# 	#change for the prospective lanes (measured from the start of that lane). Higher value is preferred. 
						# 	self.traci.vehicle.changeLane(agent_id,0,self._laneChangeAttemptDuration) 
						# else:
						# 	self.traci.vehicle.changeLane(agent_id,1, self._laneChangeAttemptDuration)
					# print("Priority Removed")
				elif action == 1:
					pass # do nothing
			else:
				if action == 0:
					self.traci.vehicle.setType(agent_id,"rl-priority")
					# self.traci.vehicle.changeLane(agent_id,0,self._laneChangeAttemptDuration) 
				elif action == 1:
					pass # do nothing
					
	
	def initSimulator(self,withGUI,portnum):
		if withGUI:
			import traci
		else:
			try:
				import libsumo as traci
			except:
				import traci
		seed = self._sumo_seed
		if self._isTestFlag == False:
			self._networkFileName = "sumo_configs/Grid1.net.xml"
			sumoConfig = "sumo_configs/sim.sumocfg"
		else:
			self._networkFileName = "sumo_configs/LargeTestNetwork.net.xml"
			sumoConfig = "sumo_configs/LargeTestNetwork.sumocfg"
   
		self.sumoCMD = ["--seed", str(seed),"--waiting-time-memory",str(self.action_steps),"--time-to-teleport", str(-1),
				 "--no-step-log","--statistic-output","output.xml"]
   
#   "--lanechange.duration",str(1),
		if withGUI:
			sumoBinary = checkBinary('sumo-gui')
			# sumoCMD += ["--start", "--quit-on-end"]
			self.sumoCMD += ["--start"]
			# self.sumoCMD += ["--start", "--quit-on-end"]
		else:	
			sumoBinary = checkBinary('sumo')

		# print(sumoBinary)
		# sumoConfig = "sumo_configs/sim.sumocfg"
		self.sumoCMD = ["-c", sumoConfig] + self.sumoCMD


		random.seed(seed)
		traci.start([sumoBinary] + self.sumoCMD)
		return traci

	def closeSimulator(traci):
		traci.close()
		sys.stdout.flush()

