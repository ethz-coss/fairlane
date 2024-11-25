from gym import Env
from gym.utils import seeding
from gym import spaces
import numpy as np
from sumolib import checkBinary
import os, sys
import traci
from scripts import utils
import xml.etree.ElementTree as ET
from itertools import combinations
import sumolib
from sumolib import net
from lxml import etree as ET
from gym.vector.utils import concatenate
from utils.common import convertToFlows, VPH
from gym.vector.utils.spaces import batch_space
import copy
rho_proximity = 15

def generate_routefile(base_routefile, out_route_file, cav_rate, hdv_rate, n_agents, baseline):
	cav_period, npc_period, hdv_period = convertToFlows(cav_rate,hdv_rate,baseline)
	# print(f'WARNING: n_agents not the same as output of convertToFlows: {n_agents} vs {_n_agents}',
	#       'Following convertToFlows')
	# print(n_agents)
	# print(_n_agents)
	# print(cav_period)
	# print(npc_period)
	# print(hdv_period)
	data = ET.Element('routes')
	base_routes = ET.parse(base_routefile)
	vehicles = base_routes.findall('vehicle')
	vtypes = base_routes.findall('vType')
	flows = base_routes.findall('flow')
	for vtype in vtypes:
		if baseline=='baseline1':
			if vtype.attrib['id']=='rl-priority':
				# vtype.attrib['maxSpeed'] = '13.89'
				vtype.attrib['vClass'] = 'passenger'
		if baseline=='baseline2':
			if vtype.attrib['id']!='cav-priority':
				# vtype.attrib['maxSpeed'] = '13.89'
				vtype.attrib['vClass'] = 'custom2'
			# if vtype.attrib['id']=='cav-priority':
			# 	vtype.attrib['vClass'] = 'custom2'
		if  baseline=='model' or  baseline=='sota':
			if vtype.attrib['id']=='rl-priority':
				# vtype.attrib['maxSpeed'] = '19.44'
				vtype.attrib['vClass'] = 'custom2'
		data.append(vtype)

	# for flow in flows:
	# 	if flow.attrib['type']=='passenger-default':
	# 		flow.attrib['period'] = f'exp({npc_period:.4f})'
	# 		if npc_period==0:
	# 			continue
	# 	if flow.attrib['type']=='cav-priority':
	# 		flow.attrib['period'] = f'exp({cav_period:.4f})'
	# 		if cav_period==0:
	# 			continue
	# 	if flow.attrib['type']=='rl-default':
	# 		flow.attrib['period'] = f'exp({hdv_period:.4f})'
	# 		if hdv_period==0:
	# 			continue
		# if 'RL' in flow.attrib['id'] and hdv_period==0:
		# 	continue
		# if flow.attrib['id']=='RL_-15_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-15_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-15_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-3_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-3_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-3_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-23_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-23_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-23_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-5_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-5_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'
		# if flow.attrib['id']=='RL_-5_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(hdv_period/3)}'

		# if 'npc' in flow.attrib['id'] and npc_period==0:
		# 	continue

		# if flow.attrib['id']=='npc_-15_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-15_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-15_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-3_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-3_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-3_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-23_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-23_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-23_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-5_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-5_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'
		# if flow.attrib['id']=='npc_-5_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(npc_period/3)}'


		# if 'cav' in flow.attrib['id'] and cav_period==0:
		# 	continue
		# if flow.attrib['id']=='cav_-15_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-15_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-3_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-3_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-23_5':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-23_3':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-5_11':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'
		# if flow.attrib['id']=='cav_-5_15':
		# 	flow.attrib['vehsPerHour'] = f'{int(cav_period/2)}'

		# data.append(flow)
	n_agent_counter = 0
	cav_agent_counter = 0
	npc_agent_counter = 0
	# # print(_n_agents)
	for i, vehicle in enumerate(vehicles):
		# if i==_n_agents:
		# 	break
		# data.append(vehicle)
		
		if vehicle.attrib['type']=='rl-default':
			if n_agent_counter < hdv_period:
				data.append(vehicle)
				n_agent_counter+=1
		if vehicle.attrib['type']=='cav-priority':
			if cav_agent_counter < cav_period:
				data.append(vehicle)
				cav_agent_counter+=1
		if vehicle.attrib['type']=='passenger-default':
			if npc_agent_counter < npc_period:
				data.append(vehicle)
				npc_agent_counter+=1
	print("Inside Route Generate")
	print(out_route_file)
	with open(out_route_file, "wb") as f:
		f.write(ET.tostring(data, pretty_print=True))

def generate_routefile_Barcelona(base_routefile, out_route_file, cav_rate, hdv_rate, n_agents, baseline, compliance=1):
	# cav_period, npc_period, hdv_period = convertToFlows(cav_rate,hdv_rate,baseline)
	# print(f'WARNING: n_agents not the same as output of convertToFlows: {n_agents} vs {_n_agents}',
	#       'Following convertToFlows')
	# print(n_agents)
	# print(_n_agents)
	# print(cav_period)
	# print(npc_period)
	# print(hdv_period)
	data = ET.Element('routes')
	base_routes = ET.parse(base_routefile)
	vehicles = base_routes.findall('vehicle')
	vtypes = base_routes.findall('vType')
	flows = base_routes.findall('flow')
	for vtype in vtypes:
		if baseline=='baseline1':
			if vtype.attrib['id']=='rl-priority':
				# vtype.attrib['maxSpeed'] = '13.89'
				vtype.attrib['vClass'] = 'passenger'
		if baseline=='baseline2':
			if vtype.attrib['id']!='cav-priority':
				# vtype.attrib['maxSpeed'] = '13.89'
				vtype.attrib['vClass'] = 'custom2'
			# if vtype.attrib['id']=='cav-priority':
			# 	vtype.attrib['vClass'] = 'custom2'
		if  baseline=='model' or  baseline=='sota':
			if vtype.attrib['id']=='rl-priority':
				# vtype.attrib['maxSpeed'] = '19.44'
				vtype.attrib['vClass'] = 'custom2'
		data.append(vtype)

	id_vehicle_list = {}
	for i, vehicle in enumerate(vehicles):
		id = vehicle.attrib['id'].split('_')[-1]
		veh_type = vehicle.attrib['type']
		id_vehicle_list.setdefault(id, {})[veh_type] = vehicle
		
	
	ids_to_delete = []
	for id, vehs in id_vehicle_list.items():
		if len(vehs)!= 3:
			ids_to_delete.append(id)
	for id in ids_to_delete:
		del id_vehicle_list[id]
	
	num_vehicles = len(id_vehicle_list)

	scale_factor = num_vehicles/VPH

	cav_period = int((num_vehicles/100)*cav_rate)
	hdv_period = int((num_vehicles/100)*hdv_rate)
	npc_period = num_vehicles - (cav_period + hdv_period)
	noncomply = 0

	noncomply = int(npc_period*(1-compliance))
	npc_period = npc_period-noncomply

	sampling = {'rl-default': hdv_period,
				'cav-priority': cav_period,
				'passenger-default': npc_period,
				'noncomply': noncomply}

	rates = np.array(list(sampling.values()))
	probabilities = rates/np.sum(rates)
	# veh_type_choices = np.random.choice(list(sampling.keys()), size=num_vehicles, replace=True, p=probabilities)
	veh_type_choices = np.array(sum([[k]*v for k, v in sampling.items()], start=[]))
	np.random.shuffle(veh_type_choices)
	for i, (id, vehs) in enumerate(id_vehicle_list.items()):
		veh_type = veh_type_choices[i]
		if veh_type=="noncomply":
			vehicle = vehs['passenger-default']
			vehicle.attrib['type'] = 'noncomply'
		else:
			vehicle = vehs[veh_type_choices[i]]
		data.append(vehicle) # choose type of vehicle randomly. indexes correspond to "same" vehicle but as 1 of 3 types

	with open(out_route_file, "wb") as f:
		f.write(ET.tostring(data, pretty_print=True))
	return scale_factor

# def generate_routefile(base_routefile, out_route_file, cav_rate, hdv_rate, n_agents, baseline):
# 	cav_period, npc_period, hdv_period = convertToFlows(cav_rate,hdv_rate,baseline)
# 	# print(f'WARNING: n_agents not the same as output of convertToFlows: {n_agents} vs {_n_agents}',
# 	#       'Following convertToFlows')
# 	# print(n_agents)
# 	# print(_n_agents)
# 	data = ET.Element('routes')
# 	base_routes = ET.parse(base_routefile)
# 	vehicles = base_routes.findall('vehicle')
# 	vtypes = base_routes.findall('vType')
# 	flows = base_routes.findall('flow')
# 	for vtype in vtypes:
# 		if baseline=='baseline1':
# 			if vtype.attrib['id']=='rl-priority':
# 				vtype.attrib['maxSpeed'] = '13.89'
# 				vtype.attrib['vClass'] = 'passenger'
# 		if baseline=='baseline2':
# 			if vtype.attrib['id']!='cav-priority':
# 				vtype.attrib['maxSpeed'] = '13.89'
# 				vtype.attrib['vClass'] = 'custom2'
# 			# if vtype.attrib['id']=='cav-priority':
# 			# 	vtype.attrib['vClass'] = 'custom2'
# 		if  baseline=='model' or  baseline=='sota':
# 			if vtype.attrib['id']=='rl-priority':
# 				vtype.attrib['maxSpeed'] = '19.44'
# 				vtype.attrib['vClass'] = 'custom2'
# 		data.append(vtype)

# 	for flow in flows:
# 		# if flow.attrib['type']=='passenger-default':
# 		# 	flow.attrib['period'] = f'exp({npc_period:.4f})'
# 		# 	if npc_period==0:
# 		# 		continue
# 		# if flow.attrib['type']=='cav-priority':
# 		# 	flow.attrib['period'] = f'exp({cav_period:.4f})'
# 		# 	if cav_period==0:
# 		# 		continue
# 		# if flow.attrib['type']=='rl-default':
# 		# 	flow.attrib['period'] = f'exp({hdv_period:.4f})'
# 		# 	if hdv_period==0:
# 		# 		continue
# 		if 'RL' in flow.attrib['id'] and hdv_period==0:
# 			continue
# 		if flow.attrib['id']=='RL_-15_3':
# 			flow.attrib['vehsPerHour'] = f'{hdv_period}'
# 		if flow.attrib['id']=='RL_-3_15':
# 			flow.attrib['vehsPerHour'] = f'{hdv_period}'
# 		if flow.attrib['id']=='RL_-23_5':
# 			flow.attrib['vehsPerHour'] = f'{hdv_period}'
# 		if flow.attrib['id']=='RL_-5_11':
# 			flow.attrib['vehsPerHour'] = f'{hdv_period}'

# 		if 'npc' in flow.attrib['id'] and npc_period==0:
# 			continue
# 		if flow.attrib['id']=='npc_-15_3':
# 			flow.attrib['vehsPerHour'] = f'{npc_period}'
# 		if flow.attrib['id']=='npc_-3_15':
# 			flow.attrib['vehsPerHour'] = f'{npc_period}'
# 		if flow.attrib['id']=='npc_-23_5':
# 			flow.attrib['vehsPerHour'] = f'{npc_period}'
# 		if flow.attrib['id']=='npc_-5_11':
# 			flow.attrib['vehsPerHour'] = f'{npc_period}'			

# 		if 'cav' in flow.attrib['id'] and cav_period==0:
# 			continue
# 		if flow.attrib['id']=='cav_-15_3':
# 			flow.attrib['vehsPerHour'] = f'{cav_period}'
# 		if flow.attrib['id']=='cav_-3_15':
# 			flow.attrib['vehsPerHour'] = f'{cav_period}'
# 		if flow.attrib['id']=='cav_-23_5':
# 			flow.attrib['vehsPerHour'] = f'{cav_period}'
# 		if flow.attrib['id']=='cav_-5_11':
# 			flow.attrib['vehsPerHour'] = f'{cav_period}'

# 		data.append(flow)
# 	# n_agent_counter = 0
# 	# cav_agent_counter = 0
# 	# npc_agent_counter = 0
# 	# # print(_n_agents)
# 	# for i, vehicle in enumerate(vehicles):
# 	# 	# if i==_n_agents:
# 	# 	# 	break
# 	# 	# data.append(vehicle)
		
# 	# 	if vehicle.attrib['type']=='rl-default':
# 	# 		if n_agent_counter < _n_agents:
# 	# 			data.append(vehicle)
# 	# 			n_agent_counter+=1
# 	# 	if vehicle.attrib['type']=='cav-priority':
# 	# 		if cav_agent_counter < cav_period:
# 	# 			data.append(vehicle)
# 	# 			cav_agent_counter+=1
# 	# 	if vehicle.attrib['type']=='passenger-default':
# 	# 		if npc_agent_counter < npc_period:
# 	# 			data.append(vehicle)
# 	# 			npc_agent_counter+=1
# 	print("Inside Route Generate")
# 	print(out_route_file)
# 	with open(out_route_file, "wb") as f:
# 		f.write(ET.tostring(data, pretty_print=True))

# def generate_routefile(base_routefile, out_route_file, cav_rate, hdv_rate, n_agents, baseline):
# 	cav_period, npc_period, _n_agents = convertToFlows(cav_rate,hdv_rate,baseline)
# 	# print(f'WARNING: n_agents not the same as output of convertToFlows: {n_agents} vs {_n_agents}',
# 	#       'Following convertToFlows')
# 	# print(n_agents)
# 	# print(_n_agents)
# 	data = ET.Element('routes')
# 	base_routes = ET.parse(base_routefile)
# 	vehicles = base_routes.findall('vehicle')
# 	vtypes = base_routes.findall('vType')
# 	flows = base_routes.findall('flow')
# 	for vtype in vtypes:
# 		if baseline=='baseline1':
# 			if vtype.attrib['id']=='rl-priority':
# 				vtype.attrib['maxSpeed'] = '13.89'
# 				vtype.attrib['vClass'] = 'passenger'
# 		if baseline=='baseline2':
# 			if vtype.attrib['id']!='cav-priority':
# 				vtype.attrib['maxSpeed'] = '13.89'
# 				vtype.attrib['vClass'] = 'custom2'
# 		if  baseline=='model' or  baseline=='sota':
# 			if vtype.attrib['id']=='rl-priority':
# 				vtype.attrib['maxSpeed'] = '19.44'
# 				vtype.attrib['vClass'] = 'custom2'
# 		data.append(vtype)

# 	# for flow in flows:
# 	# 	if flow.attrib['type']=='passenger-default':
# 	# 		flow.attrib['period'] = f'exp({npc_period:.4f})'
# 	# 		if npc_period==0:
# 	# 			continue
# 	# 	if flow.attrib['type']=='cav-priority':
# 	# 		flow.attrib['period'] = f'exp({cav_period:.4f})'
# 	# 		if cav_period==0:
# 	# 			continue
# 	# 	data.append(flow)
# 	n_agent_counter = 0
# 	cav_agent_counter = 0
# 	npc_agent_counter = 0
# 	# print(_n_agents)
# 	for i, vehicle in enumerate(vehicles):
# 		# if i==_n_agents:
# 		# 	break
# 		# data.append(vehicle)
		
# 		if vehicle.attrib['type']=='rl-default':
# 			if n_agent_counter < _n_agents:
# 				data.append(vehicle)
# 				n_agent_counter+=1
# 		if vehicle.attrib['type']=='cav-priority':
# 			if cav_agent_counter < cav_period:
# 				data.append(vehicle)
# 				cav_agent_counter+=1
# 		if vehicle.attrib['type']=='passenger-default':
# 			if npc_agent_counter < npc_period:
# 				data.append(vehicle)
# 				npc_agent_counter+=1

# 	with open(out_route_file, "wb") as f:
# 		f.write(ET.tostring(data, pretty_print=True))

class Agent:
    def __init__(self, env, n_agent, edge_agent=None):
        """Dummy agent object"""
        self.edge_agent = edge_agent
        self.env = env

        self.id = n_agent
        self.name = f'{self.id}'

class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	
	def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',testStatAccumulation=10,
				 testFlag='False',testModel="Default",simulation_end=36000, num_agents=50, action_step=30,
				 episode_duration=None, cav_rate=10, hdv_rate=50, scenario_flag='model',
				 default_seed=42, waiting_time_memory=3, compliance=1):
		self.pid = os.getpid()
		self.seed(default_seed)
		self.waiting_time_memory = waiting_time_memory
		self.sumoCMD = []
		self.reset_counter = 0
		self._simulation_end = simulation_end
		self._mode = mode
		self._testModel = testModel
		self.SotaFlag = scenario_flag=='sota'
		self.scenario_flag = scenario_flag
		self._testStatAccumulation = testStatAccumulation
		self.agents = []
		self.scaleFactor = 1
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
			if testModel=="Default":
				self._networkFileName = "sumo_configs/Test/MSN_Grid_rebuildTrafficLight.net.xml"
				self._baseRouteFileName = "sumo_configs/Test/MSN_Grid_base.rou.xml"
				self._routeFileName = f"sumo_configs/Test/rou_{cav_rate}_{hdv_rate}_{scenario_flag}.rou.xml"
				generate_routefile(self._baseRouteFileName, self._routeFileName, cav_rate, hdv_rate, num_agents, scenario_flag)
				self._warmup_steps = 300
				self.sumoConfig = "sumo_configs/Test/MSN_Grid.sumocfg"
			elif testModel=="Barcelona":
				self._networkFileName = "sumo_configs/Test/Barcelona/Barcelona.net.xml"
				self._baseRouteFileName = "sumo_configs/Test/Barcelona/Barcelona_scaled.rou.xml"
				self._routeFileName = f"sumo_configs/Test/Barcelona/routes/rou_{cav_rate}_{hdv_rate}_{scenario_flag}_{default_seed}_{compliance}.rou.xml"
				scalefactor = generate_routefile_Barcelona(self._baseRouteFileName, self._routeFileName, cav_rate, hdv_rate, num_agents, scenario_flag, compliance=compliance)
				self.scaleFactor = 1/scalefactor
				self._warmup_steps = 300
				self.sumoConfig = "sumo_configs/Test/Barcelona/Barcelona.sumocfg"

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
		self._emergencyBreaking = 0
		self._collisionVehicleID = []
		self._agentModelDict = {}
		self._reward_type = "Global" 
		# self._reward_type = "Local" 
		# self._reward_type = "Individual"
		self.withGUI = mode
		self.action_steps = action_step
		self.episode_duration = episode_duration
		self._nextStepobs = []
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
		self._avg_speed_npc = 0
		self._avg_speed_heuristic = 0
		self._avg_speed_npc = 0
		self._avg_speed_cav = 0		
		self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0
		self._departDelay_rl = 0;self._departDelay_npc = 0;self._departDelay_cav = 0
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
		
		

		self.traci = self.initSimulator(self.withGUI, self.pid)
		self.resetAllVariables()
		if testFlag:
			self.agents = self.createNDynamicAgents()
		else:
			self.agents = self.createNAgents()

	@property
	def controlled_vehicles(self):
		return self.agents
	
	def createNAgents(self):
		agents = [Agent(self, f'RL_{i}') for i in range(self.n)]
		self._agentModelDict = {agent.name: agent for agent in agents}
		return agents
	
	def createNDynamicAgents(self):
		allVehicleList = self.all_vehicles
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
		
		for rl in self._rl_vehicleID:
			if rl not in self._agentModelDict: # add all new RL agents
				agent = Agent(self, rl)
				self._agentModelDict[rl] = agent
				self.agents.append(agent)
						
		for key in list(self._agentModelDict.keys()):
			if not key in self._rl_vehicleID:
				self.agents.remove(self._agentModelDict[key]) 
				del self._agentModelDict[key]
		# if len(self._rl_vehicleID) < 1:
		# 	self.n=86
		# else:
		# 	self.n = len(self._rl_vehicleID)
		# print(len(self.agents))
		# print(self.n)
		assert(len(self.agents)<=self.n, len(self.agents))
		return self.agents

	
	def checkIfAgentExist(self,agent_id):
		allVehicleList = self.all_vehicles
		if agent_id in allVehicleList:
			return True
		else:
			return False

	def checkIfTeleport(self,agent_id):
		try:
			listt = self.traci.vehicle.getNextLinks(agent_id)
			if len(listt)==0:
				# print("RL Agent does not exist. Adding dummy State for :" + str(agent_id))
				return True
			else:
				return False
		except traci.libsumo.TraCIException:
			return True
		
	
	
	def getState(self,agent_id):
		"""
		Retrieve the state of the network from sumo. 
		"""
		state = []
		localRLVehicleCount = 0
		allvehicleLocalcounter = 1
		
		# print(agent_id)
		if self.checkIfAgentExist(agent_id)==False:
			state = [0,0,0,0,0,0,0]
			print(agent_id,"--does not exist")		
		else:
			listt = self.traci.vehicle.getNextLinks(agent_id)
			edge_id = self.traci.vehicle.getRoadID(agent_id)
			lane_id = self.traci.vehicle.getLaneID(agent_id)
			isPriorityLane = 1
			if lane_id.find("_0")==-1:
				isPriorityLane = 0
			
			# allVehicleList = self.all_vehicles # TODO: double check: DAMIAN
			if len(listt) < 1:
				nextLane = lane_id # hack 
			else:
				nextLane = listt[0][0]
			self._nextLane[agent_id] = nextLane
			priorityLane_id = edge_id + "_0"
			routeList = self.traci.vehicle.getRoute(agent_id)
			nextPriorityLane_id = nextLane.split("_")[0] + "_0"
			#check occupancy of that lane
			# occupancyNextLane = self.traci.lane.getLastStepOccupancy(nextLane)
			# occupancyCurrentLane = self.traci.lane.getLastStepOccupancy(lane_id)
			# occupancyCurrentEdge = self.traci.edge.getLastStepOccupancy(edge_id)
			# occupancyCurrentPriorityLane = self.traci.lane.getLastStepOccupancy(priorityLane_id)
			# occupancyNextPriorityLane = self.traci.lane.getLastStepOccupancy(nextPriorityLane_id)
	
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
				elif priority_type!="cav-priority":
					lane_id = str(self.traci.vehicle.getLaneIndex(cav))
					if lane_id == '1': # CAV lane is always 0
						npc_lane_position = self.traci.vehicle.getLanePosition(cav)
						diff = agent_lane_pos - npc_lane_position
						if abs(diff)<=rho_proximity:
							npcCount+=1
						diff = npc_lane_position - agent_lane_pos
						if npc_lane_position>=agent_lane_pos:
							if abs(diff)<=rho_proximity:
								npcCount+=1

		
			#find next edge in the route for the RL agent

			
			# #traffic light phase related observations
			# if self._testModel=="Barcelona":
			# 	phaseState = 0
			# 	remainingDuration = 0


			remainingDuration = 1
			if priorityLane_id.find(":")!=-1 or lane_id.find(":")!=-1:
				phaseState = 0
			else: 
				if priorityLane_id in self.lane2tls:
					tls_id = self.lane2tls[priorityLane_id]
					# lanesList = self.traci.trafficlight.getControlledLanes(nextNodeID)
					lanesListLink = self.traci.trafficlight.getControlledLinks(tls_id)
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
							

					trafficState = self.traci.trafficlight.getRedYellowGreenState(tls_id)
					phaseDuration = self.traci.trafficlight.getPhaseDuration(tls_id)
					remainingPhaseDuration = self.traci.trafficlight.getNextSwitch(tls_id) - self.traci.simulation.getTime()
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
				else:
					phaseState = 0

			self._trafficPhaseRLagent[agent_id] = phaseState
			trafficState = self.traci.vehicle.getNextTLS(agent_id)
			isStopped = 0
			speed = self.all_veh_speeds[agent_id]
		
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
		allVehicleList = self.all_vehicles
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
		self._avg_speed_npc = 0
		self._avg_speed_heuristic = 0
		self._avg_speed_npc = 0
		self._avg_speed_cav = 0
		self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0
		self._departDelay_rl = 0;self._departDelay_npc = 0;self._departDelay_cav = 0
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
		self._emergencyBreaking = 0

		self.all_vehicles = []
		self.all_veh_speeds = {}
		self._laneList = []

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
	
	# def setHeuristicAgentTogglePriority(self): #heuristic logic to change priority of all agent starting with heuristic in the name. 
		#this is done to overcome the limitation of training 100's of RL agent. Can we just train 25% of the RL agent with heuristic logic and 
		#still get similar or better training output? One novelty of the paper, probably?
		# allVehicleList = self.all_vehicles # TODO: double check: DAMIAN # TODO: FIX DAMIAN
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
						# speed = self.all_veh_speeds[heuristic]
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
						# speed = self.all_veh_speeds[heuristic]
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
		allVehicleList = self.all_vehicles
		lane_list = self._laneList
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
				if lane_01 not in lane_list:
					continue # move to next rl vehicle
				vehicle_on_priority_lane = self.traci.lane.getLastStepVehicleIDs(priority_lane)
				npc_vehicleID,rl_vehicleID, heuristic_vehicleID,cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(vehicle_on_priority_lane)
				heuristic_lane_position = self.traci.vehicle.getLanePosition(rl)
				vehicle_on_lane = self.traci.lane.getLastStepVehicleIDs(lane_01)
				for veh in vehicle_on_lane:
					npc_lane_position = self.traci.vehicle.getLanePosition(veh)
					diff = heuristic_lane_position - npc_lane_position 
					if abs(diff)<= 0.5: #and heuristic_lane_position>npc_lane_position: 
						npcCount+=1
						break
					
					# if heuristic_lane_position<npc_lane_position:
					# 	diff = npc_lane_position - heuristic_lane_position
					# 	if diff<=0.5:
					# 		npcCount+=1 
				
				flag = False
				diff = 0
				cav_lane_position = -999

				# for rl_temp in rl_vehicleID:					
				# 	cav_vehicleID.append(rl_temp) # added to consider heavy HDV flows. 
				for cav in cav_vehicleID:
					cav_lane_position = self.traci.vehicle.getLanePosition(cav)
					diff = heuristic_lane_position - cav_lane_position
					if diff>= self._lane_clearing_distance_threshold:					
						continue
					else:
						#change priority of heuristic agent as it is inside clearing distance
						# speed = self.all_veh_speeds[heuristic]
						if diff > 0 and npcCount==0:
						# if diff > 0:
							# print(str(diff),"--",str(heuristic_lane_position),"--",str(cav_lane_position))
							self._collisionCounter+=1
							self.traci.vehicle.setType(rl,"rl-default")
							lane_index = self.traci.vehicle.getLaneIndex(rl)
							# if lane_index==0:
							# 	self.traci.vehicle.changeLane(rl,1,0)
							flag = True
							break
				if flag==False: 
					# if lane_index!=0 and self.traci.vehicle.getTypeID(rl)!="rl-priority":
					if lane_index!=0:
						# speed = self.all_veh_speeds[heuristic]
						# if speed>0.2:
						self.traci.vehicle.setType(rl,"rl-priority")
						lane_index = self.traci.vehicle.getLaneIndex(rl)
						# if lane_index!=0:
						# 	self.traci.vehicle.changeLane(rl,0,0)
							
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
		self.reset_counter += 1
		# print("Collision_Counter :" + str(self._collisionCount))
		self._sumo_step = 0
		obs_n = []
		seed = self._sumo_seed #+ self.reset_counter*100
		self.traci.load(self.sumoCMD + ["--seed", str(seed)])
		self.resetAllVariables()
		self._laneList = self.traci.lane.getIDList()
		#WARMUP PERIOD
		while self._sumo_step < self._warmup_steps:
			self.traci.simulationStep() 		# Take a simulation step to initialize
			self.all_vehicles = self.traci.vehicle.getIDList()
			self.all_veh_speeds = {veh_id: self.traci.vehicle.getSpeed(veh_id) for veh_id in self.all_vehicles}

			# print(self.traci.vehicle.getTimeLoss("RL_9"))
			# if self._sumo_step == 10:
			# 	allVehicleList = self.all_vehicles # TODO: double check: DAMIAN
			# 	self._npc_vehicleID,self.original_rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
				
				# self.initializeRLAgentStartValues()
			# 	self.keepRLAgentLooping()
			# if self._sumo_step%self.action_steps==0:
			# 	# self.setHeuristicAgentTogglePriority()
			if self.scenario_flag not in ['baseline1','baseline2']:
				self.setRLAgentTogglePriority()
			self._sumo_step +=1

		#record observatinos for each agent
		self.agents = self.createNDynamicAgents()
		# self.initializeNPCRandomPriority()
		for agent in self.agents:
			agent.done = False
			obs_n.append(self._get_obs(agent))
		while len(obs_n)<self.n:
			obs_n.append([0,0,0,0,0,0,0])

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
		return self.getState(f'{agent.id}')
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
			speed = self.all_veh_speeds[rl_agent]
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
	
	def computeNPCProximityReward(self,rl_agent):
		reward = +0.5
		rl_lane_id = self.traci.vehicle.getLaneID(rl_agent)
		edge = rl_lane_id.split("_")[0]
		rl_lane_index = rl_lane_id.split("_")[1]
		npcCount = 0
		if rl_lane_index=="0":
			rl_priority_type = self.traci.vehicle.getTypeID(rl_agent)
			if self.traci.vehicle.getTypeID(rl_agent)=="rl-default":		
				lane_1 = edge + "_1"		
				vehList = self.traci.lane.getLastStepVehicleIDs(lane_1)
				for veh in vehList:
					npc_lane_position = self.traci.vehicle.getLanePosition(veh)
					agent_lane_pos = self.traci.vehicle.getLanePosition(rl_agent)
					diff = agent_lane_pos - npc_lane_position
					if abs(diff)<=rho_proximity:
						npcCount+=1
					diff = npc_lane_position - agent_lane_pos
					if npc_lane_position>=agent_lane_pos:
						if abs(diff)<=rho_proximity:
							npcCount+=1
			if npcCount>0:
				reward = -0.5
		return reward
	
	# get reward for a particular agent
	def _get_reward(self,agent):
		agent_id = f'{agent.id}'
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
				reward_npc_proximity = self.computeNPCProximityReward(agent_id)
				reward_dist_intersection = self.computeRLRewardDistFromIntersection(agent_id)
				
				# print("reward: " + str(reward_cav_priority))
				# print("-------------------------")
				# reward_priority_lane_Speed = self.computeAvgSpeedPriorityLaneReward(agent_id)
				# overall_reward = reward_cooperative + reward_overallNetwork + reward_cav_priority
				# overall_reward = reward_cav_priority
				# print(overall_reward)		
				# overall_reward = self._weightCAVPriority*reward_cav_priority + self._weightRLWeightingTime*reward_RLWaitingTime + self._weightCAVWeightingTime*reward_cavWaitingTime
				# overall_reward = reward_cav_priority + 0.2*reward_RLWaitingTime + 0.2*reward_cavWaitingTime
				overall_reward = reward_dist_intersection + reward_cav_priority + reward_speed_RL + reward_speed_CAV + reward_npc_proximity
				# overall_reward = reward_cav_priority
		return overall_reward
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		np.random.seed(seed)
		self._sumo_seed = seed
		return [seed]

	def _get_done(self, agent):
		if (self.traci.simulation.getTime() > (self.episode_duration + self._warmup_steps)) and self._isTestFlag==False:
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
			speed = self.all_veh_speeds[veh]
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
		
	# 	allVehicleList = self.all_vehicles # TODO: double check: DAMIAN
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
		
		allVehicleList = self.all_vehicles
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

				self._BeforeSpeed[rl_agent] = self.all_veh_speeds[rl_agent]
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
							speed = self.all_veh_speeds[cav]
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
							total_cav_speed+= speed
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
			
			# print(self._observationCounterAfterAction)
			for rl_agent in self._rl_vehicleID:
				if self.checkIfTeleport(rl_agent):
					continue
				accumulated_time_loss = 0
				total_waiting_time=0
				total_waiting_time_cav=0
				total_cav_speed = 0
				self._AfterSpeed[rl_agent] = self.all_veh_speeds[rl_agent]
				self._afterPriorityForRLAgent[rl_agent] = self.traci.vehicle.getTypeID(rl_agent)
				lane_id = self.traci.vehicle.getLaneID(rl_agent)
				for veh in self._listOfVehicleIdsInConcern.get(rl_agent, []):
					if veh in allVehicleList:
						# elapsed_vehicle_time = self.traci.vehicle.getDeparture(veh)
						# accumulated_time_loss+=self.traci.vehicle.getTimeLoss(veh)/(elapsed_simulation_time - elapsed_vehicle_time)
						veh_wait_time = self.traci.vehicle.getAccumulatedWaitingTime(veh)
						total_waiting_time+=veh_wait_time
						priority_type = self.traci.vehicle.getTypeID(veh)
						if priority_type=="cav-priority": 
							cav_lane_position = self.traci.vehicle.getLanePosition(veh)
							# if self.traci.vehicle.isStopped(veh)==True:
							# 	g = 0
							# 	edge_length = self.traci.lane.getLength(lane_id)
							# 	distanceFromIntersection = edge_length - cav_lane_position
							cav_wait_time = veh_wait_time
							# if cav_wait_time > 2:
							# 	g = 0
							total_waiting_time_cav+=cav_wait_time
							total_cav_speed += self.all_veh_speeds[veh]

				
			
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

			#CAV related measures
			avg_speed_cav = 0;currentWaitingTime_cav=0;departDelay_cav=0;departDelay_rl=0;departDelay_npc=0
			cav_counter = 0
			for cav_agent in self._cav_vehicleID:
				cav_edge = self.traci.vehicle.getRoadID(cav_agent)
				# if cav_edge=="-15" or cav_edge=="-5" or cav_edge=="-23" or cav_edge=="-3":
				cav_counter+=1
				avg_speed_cav+=self.all_veh_speeds[cav_agent]
				# print(self.traci.vehicle.getAccumulatedWaitingTime(cav_agent))
				# assert(self.traci.vehicle.getAccumulatedWaitingTime(cav_agent)<3)
				currentWaitingTime_cav += self.traci.vehicle.getAccumulatedWaitingTime(cav_agent)
				departDelay_cav += self.traci.vehicle.getDepartDelay(cav_agent)
			if cav_counter > 0:
				self._avg_speed_cav += avg_speed_cav/cav_counter
				# print(currentWaitingTime_cav/cav_counter)
				self._currentWaitingTime_cav += currentWaitingTime_cav/cav_counter
				self._departDelay_cav += departDelay_cav/cav_counter

				if self._currentWaitingTime_cav>=360000:
					stophere = 0
				assert(self._currentWaitingTime_cav<=360000)	
			else:
				self._avg_speed_cav += 0
				self._currentWaitingTime_cav += 0
				self._departDelay_cav += 0
			self._cavFlowCounter +=cav_counter
			# print(self._currentWaitingTime_cav)
			# print(cav_counter)

			avg_speed_rl=0;currentWaitingTime_rl=0
			total_lane_change_all_RL = 0
			total_lane_change_all = 0
			rl_counter = 0


			#RL related stats
			for rl_agent in self._rl_vehicleID:
				rledge = self.traci.vehicle.getRoadID(rl_agent)
				# if rledge=="-15" or rledge=="-5" or rledge=="-23" or rledge=="-3":
				rl_counter+=1
				avg_speed_rl+=self.all_veh_speeds[rl_agent]
				currentWaitingTime_rl += self.traci.vehicle.getAccumulatedWaitingTime(rl_agent)
				departDelay_rl += self.traci.vehicle.getDepartDelay(rl_agent)

			if rl_counter > 0:
				self._avg_speed_rl += avg_speed_rl/rl_counter
				self._currentWaitingTime_rl += currentWaitingTime_rl/rl_counter
				self._departDelay_rl += departDelay_rl/rl_counter
			else:
				self._avg_speed_rl += 0
				self._currentWaitingTime_rl += 0
				self._departDelay_rl += 0

		
		
			#NPC related measures
			avg_speed_npc=0;currentWaitingTime_npc=0;npc_counter = 0
			for npc_agent in self._npc_vehicleID:
				npcedge = self.traci.vehicle.getRoadID(npc_agent)
				# if npcedge=="-15" or npcedge=="-5" or npcedge=="-23" or npcedge=="-3":
				npc_counter+=1
				avg_speed_npc+=self.all_veh_speeds[npc_agent]
				currentWaitingTime_npc += self.traci.vehicle.getAccumulatedWaitingTime(npc_agent)
				departDelay_npc += self.traci.vehicle.getDepartDelay(npc_agent)

			if npc_counter > 0:
				self._avg_speed_npc += avg_speed_npc/npc_counter
				self._currentWaitingTime_npc += currentWaitingTime_npc/npc_counter
				self._departDelay_npc += departDelay_npc/npc_counter
			else:
				self._avg_speed_npc += 0
				self._currentWaitingTime_npc += 0
				self._departDelay_npc += 0
		
		# print(self._currentWaitingTime_cav)
     
	def collectObservationPerStep(self):
		elapsed_simulation_time = self.traci.simulation.getTime()
		allVehicleList = self.all_vehicles
		self._emergencyBreaking+= self.traci.simulation.getEmergencyStoppingVehiclesNumber()
		self._collisionCount+= self.traci.simulation.getCollidingVehiclesNumber()
		# self._collisionVehicleID.append(self.traci.simulation.getEmergencyStoppingVehiclesIDList())
		# self._collisionCount+= self.traci.simulation.getEndingTeleportNumber()

		# self._collisionVehicleID.append(self.traci.simulation.getEmergencyStoppingVehiclesIDList())
		
		# self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle = self.getSplitVehiclesList(allVehicleList)
		
				
		# ld="det_-15_0_1_passenger"
		# temp_list = self.traci.inductionloop.getIntervalVehicleIDs(ld)
		# lane_id = self.traci.vehicle.getLaneID("RL_218")
		# for v in temp_list:
		# 	if v.find("cav")!=-1:
		# 		self._cavFlowCounter+= self.traci.inductionloop.getIntervalVehicleNumber(ld)

		# avg_speed_rl=0;currentWaitingTime_rl=0
		# total_lane_change_all_RL = 0
		# total_lane_change_all = 0
		# rl_counter = 0


		# #RL related stats
		# for rl_agent in self._rl_vehicleID:
		# 	rledge = self.traci.vehicle.getRoadID(rl_agent)
		# 	if rledge=="-15" or rledge=="-5" or rledge=="-23" or rledge=="-3":
		# 		rl_counter+=1
		# 		avg_speed_rl+=self.all_veh_speeds[rl_agent]
		# 		currentWaitingTime_rl += self.traci.vehicle.getWaitingTime(rl_agent)

		# if rl_counter > 0:
		# 	self._avg_speed_rl += avg_speed_rl/rl_counter
		# 	self._currentWaitingTime_rl += currentWaitingTime_rl/rl_counter
		# else:
		# 	self._avg_speed_rl += 0
		# 	self._currentWaitingTime_rl += 0

	
	
		# #NPC related measures
		# avg_speed_npc=0;currentWaitingTime_npc=0;npc_counter = 0
		# for npc_agent in self._npc_vehicleID:
		# 	npcedge = self.traci.vehicle.getRoadID(npc_agent)
		# 	if npcedge=="-15" or npcedge=="-5" or npcedge=="-23" or npcedge=="-3":
		# 		npc_counter+=1
		# 		avg_speed_npc+=self.all_veh_speeds[npc_agent]
		# 		currentWaitingTime_npc += self.traci.vehicle.getWaitingTime(npc_agent)

		# if npc_counter > 0:
		# 	self._avg_speed_npc += avg_speed_npc/npc_counter
		# 	self._currentWaitingTime_npc += currentWaitingTime_npc/npc_counter
		# else:
		# 	self._avg_speed_npc += 0
		# 	self._currentWaitingTime_npc += 0

		# #CAV related measures
		# avg_speed_cav = 0;currentWaitingTime_cav=0
		# cav_counter = 0
		# for cav_agent in self._cav_vehicleID:
		# 	cav_edge = self.traci.vehicle.getRoadID(cav_agent)
		# 	if cav_edge=="-15" or cav_edge=="-5" or cav_edge=="-23" or cav_edge=="-3":
		# 		cav_counter+=1
		# 		avg_speed_cav+=self.all_veh_speeds[cav_agent]
		# 		print(self.traci.vehicle.getWaitingTime(cav_agent))
		# 		currentWaitingTime_cav += self.traci.vehicle.getWaitingTime(cav_agent)
		# if cav_counter > 0:
		# 	self._avg_speed_cav += avg_speed_cav/cav_counter
		# 	self._currentWaitingTime_cav += currentWaitingTime_cav/cav_counter
			
		# else:
		# 	self._avg_speed_cav += 0
		# 	self._currentWaitingTime_cav += 0

		


	def getTestStats(self):
	
		# print(self.traci.simulation.getTime())
		# if  self._episodeStep!=249: # to avoid writing the last line of CSV. 
		avg_delay_RL=0;avg_speed_RL=0;avg_delay_NPC=0;avg_speed_NPC=0;avg_occupancy_priorityLane=0;avg_PMx_emission=0;total_lane_change_number=0
		avg_delay_CAV=0;avg_speed_CAV=0;avg_delay_Heuristic=0;avg_speed_Heuristic=0;avg_delay_ALLButCAV=0;avg_speed_AllButCAV=0
		avg_departDelay_CAV = 0;avg_departDelay_NPC = 0;avg_departDelay_RL = 0
		
		# avg_delay_CAV = self._currentTimeLoss_cav/self.action_steps
		avg_delay_CAV = self._currentWaitingTime_cav/100
		# print("CAV_Delay--",avg_delay_CAV,"CAV Counter--",self._cavFlowCounter)
		allVehicleList = self.all_vehicles
		# print(len(allVehicleList))
		avg_speed_CAV = self._avg_speed_cav/100

		avg_departDelay_CAV = self._departDelay_cav/100

		# avg_delay_RL = self._currentTimeLoss_rl/self.action_steps
		avg_delay_RL = self._currentWaitingTime_rl/100
		avg_speed_RL = self._avg_speed_rl/100
		avg_departDelay_RL = self._departDelay_rl/100


		avg_delay_NPC = self._currentWaitingTime_npc/100
		avg_departDelay_NPC = self._departDelay_npc/100

		avg_speed_NPC = self._avg_speed_npc/100
		# if avg_delay_NPC!=0 and avg_delay_RL!=0:
		# 	total_avg_delay = (avg_delay_RL + avg_delay_NPC)/2
		# 	total_Avg_speed = (avg_speed_RL + avg_speed_NPC)/2
		# if avg_delay_NPC==0 and avg_delay_RL!=0:
		# 	total_avg_delay = avg_delay_RL
		# 	total_Avg_speed = avg_speed_RL
		# if avg_delay_NPC!=0 and avg_delay_RL==0:
		# 	total_avg_delay = avg_delay_NPC
		# 	total_Avg_speed = avg_speed_NPC
		

		allVehicleList = self.all_vehicles
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)

		cav_count = len(self._cav_vehicleID)
		npc_count = len(self._npc_vehicleID)
		rl_count = len(self._rl_vehicleID)
		if (rl_count+npc_count) > 0:
			total_avg_delay = (rl_count*avg_delay_RL + npc_count*avg_delay_NPC)/(rl_count+npc_count)
			total_Avg_speed = (rl_count*avg_speed_RL + npc_count*avg_speed_NPC)/(rl_count+npc_count)
			total_avg_departDelay = (rl_count*avg_departDelay_RL + npc_count*avg_departDelay_NPC)/(rl_count+npc_count)
		else:
			total_avg_delay = (rl_count*avg_delay_RL + npc_count*avg_delay_NPC)
			total_Avg_speed = (rl_count*avg_speed_RL + npc_count*avg_speed_NPC)
			total_avg_departDelay = (rl_count*avg_departDelay_RL + npc_count*avg_departDelay_NPC)


		# print(self._collisionCounter)
		#throughput computation using loop detector
		throughput = 0
		all_LD = self.traci.inductionloop.getIDList()
		LD_counter=0
		for ld in all_LD:			
			# if ld=="det_-15_0_1_passenger" or ld=="det_-15_1_1_passenger" or ld=="det_-15_2_1_passenger" or \
			# 	ld=="det_-23_0_1_passenger" or ld=="det_-23_1_1_passenger" or ld=="det_-23_2_1_passenger" or \
			# 	ld=="det_-3_0_1_passenger" or ld=="det_-3_1_1_passenger" or ld=="det_-3_2_1_passenger" or \
			# 	ld=="det_-5_0_1_passenger" or ld=="det_-5_1_1_passenger" or ld=="det_-5_2_1_passenger":
				LD_counter+=1
				throughput += self.traci.inductionloop.getLastIntervalVehicleNumber(ld)	#it was getlastInterval. check	
			# LD_counter+=1
			# throughput += self.traci.inductionloop.getLastIntervalVehicleNumber(ld)
			
		# ld =="det_-3_0_1_passenger" or ld=="det_-3_1_1_passenger" or ld =="det_-3_2_1_passenger" or 
		if LD_counter > 0:
			self._average_throughput = throughput/LD_counter

		avg_throughput = (self._average_throughput*3600)/(300) #vehicle/hour

		if cav_count+rl_count+npc_count > 0:
			average_delay_All = (cav_count*avg_delay_CAV+rl_count*avg_delay_RL + npc_count*avg_delay_NPC)/(cav_count+rl_count+npc_count)
			average_speed_All = (cav_count*avg_speed_CAV+rl_count*avg_speed_RL + npc_count*avg_speed_NPC)/(cav_count+rl_count+npc_count)
			average_departDelay_ALL = (cav_count*avg_departDelay_CAV + rl_count*avg_departDelay_RL + npc_count*avg_departDelay_NPC)/(cav_count+rl_count+npc_count)
		else:
			average_delay_All = (cav_count*avg_delay_CAV+rl_count*avg_delay_RL + npc_count*avg_delay_NPC)
			average_speed_All = (cav_count*avg_speed_CAV+rl_count*avg_speed_RL + npc_count*avg_speed_NPC)
			average_departDelay_ALL = (cav_count*avg_departDelay_CAV + rl_count*avg_departDelay_RL + npc_count*avg_departDelay_NPC)

		# print(avg_throughput)
		headers = ['Average Depart Delay (CAV)','Average Depart Delay (SAV & HDV)','Average Depart Delay (ALL)','Average Waiting Time (All)','Average Speed (All)','Average Waiting Time (CAV)','Average Waiting Time (SAV & NPC)','Average Speed (CAV)','Average Speed (SAV & NPC)','Throughput','Average Lane Change Number (All)','Average Lane Change Number (SAV)','Collision Number','Emergency Breaking','Episode_Step',"Seed"]
		values = [avg_departDelay_CAV,total_avg_departDelay,average_departDelay_ALL,average_delay_All,average_speed_All,avg_delay_CAV,total_avg_delay,avg_speed_CAV,total_Avg_speed,avg_throughput,self._average_LaneChange_number_all,self._average_LaneChange_number_rl,self._collisionCount/300,self._emergencyBreaking/300,self._episodeStep,self._sumo_seed]
		
		self._currentTimeLoss_cav=0;self._avg_speed_cav=0;self._currentTimeLoss_rl=0;self._avg_speed_rl=0;self._currentTimeLoss_npc=0;self._avg_speed_npc=0
		self._average_priorityLane_occupancy=0;self._average_throughput=0;self._average_PMx_emission=0;self._currentTimeLoss_Heuristic=0;self._avg_speed_heuristic=0;self._average_LaneChange_number_all=0;self._average_LaneChange_number_rl=0
		self._currentWaitingTime_Heuristic=0;self._currentWaitingTime_rl=0;self._currentWaitingTime_npc=0;self._currentWaitingTime_cav=0;self._average_LaneChange_number=0;self._collisionCounter=0;self._collisionCount=0;self._emergencyBreaking=0
		self._avg_speed_npc=0;self._currentWaitingTime_npc=0;self._cavFlowCounter=0;self._departDelay_cav=0;self._departDelay_rl=0;self._departDelay_npc=0
		# print("resetting the waiting time")
		return headers, values
	
	def make_action(self,actions):
		# assumes actions are one-hot encoded
		agent_actions = []
		# for i in range(0,self.n): 
		for i in range(len(self.agents)): 
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
		
		

		self._episodeStep+=3 # it was 1. Made it to 3 for teststat logs. Might break train
		self._sumo_step = 0
		
		if actionFlag == True:
			temp_action_dict = {}
			simple_actions = self.make_action(action_n)
			# print(simple_actions)
			for i, agent in enumerate(self.agents):
				self.lastActionDict[f'{agent.id}'] = simple_actions[i]
			
			self.collectObservation(True)		#Observation before taking an action - lastTimeStepFlag
			self._set_action()			
			# print(simple_actions)
			actionFlag = False
		
		allVehicleList = self.all_vehicles
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
		while self._sumo_step < self.action_steps:
			# advance world state
			self.collectObservationPerStep()
			self.traci.simulationStep()
			self._sumo_step +=1	
			
			# self.collectObservation(False) ##Observation at each step till the end of the action step count (for reward computation) - lastTimeStepFlag lastTimeStepFlag
			# self.keepRLAgentLooping()
			# for loop in self.traci.inductionloop.getIDList():
			# 	vehicleCount +=self.traci.inductionloop.getLastStepVehicleNumber(loop)
		
		self.all_vehicles = self.traci.vehicle.getIDList()
		self.all_veh_speeds = {veh_id: self.traci.vehicle.getSpeed(veh_id) for veh_id in self.all_vehicles}

		# print("Average Number of Vehicles per edge in five minutes are :",vehicleCount/len(self.traci.inductionloop.getIDList()))
		# print(len(self.traci.inductionloop.getIDList()))
		total_lane_change_all = 0
		total_lane_change_rl = 0
		self.collectObservation(False) #lastTimeStepFlag
		allVehicleList = self.all_vehicles
		self._npc_vehicleID,self._rl_vehicleID,self._heuristic_vehicleID,self._cav_vehicleID,ratioOfHaltVehicle= self.getSplitVehiclesList(allVehicleList)
		
		for veh in allVehicleList:
			# vehedge = self.traci.vehicle.getRoadID(veh)
			# if vehedge in ["-15", "-5", "-23", "-3"]:
			if veh in self._allVehLaneIDBefore:
				if self._allVehLaneIDBefore[veh] != self.traci.vehicle.getLaneID(veh):
					total_lane_change_all+=1
		if len(self._allVehLaneIDBefore) > 0:
			self._average_LaneChange_number_all+= total_lane_change_all/len(self._allVehLaneIDBefore)
		else:
			self._average_LaneChange_number_all+= total_lane_change_all

		for rl_agent in self._rl_vehicleID:			
			vehedge = self.traci.vehicle.getRoadID(rl_agent)
			if rl_agent in self._rlLaneID:
				# if vehedge=="-15" or vehedge=="-5" or vehedge=="-23" or vehedge=="-3":
				if self._rlLaneID[rl_agent] != self.traci.vehicle.getLaneID(rl_agent):
					total_lane_change_rl+=1

		self._average_LaneChange_number_rl+= total_lane_change_rl/len(self._rl_vehicleID) if self._rl_vehicleID else 0 
		# print("Total npc: " + str(len(self._npc_vehicleID)) + "Total RL agent: " + str(len(self._rl_vehicleID)))
		
		# if len(self._rl_vehicleID)!=self.n:
		# 	print("Total RL agent before loadState: " + str(len(self._rl_vehicleID)))
		# 	self.traci.simulation.loadState('sumo_configs/savedstate.xml')
		# 	print("Total RL agent After loadState: " + str(len(self._rl_vehicleID)))
			# self.keepRLAgentLooping()


			
		# allVehicleList = self.all_vehicles # TODO: double check: DAMIAN
		# self._npc_vehicleID,self._rl_vehicleID = self.getSplitVehiclesList(allVehicleList)
		# print("Total npc: " + str(len(self._npc_vehicleID)) + "Total RL agent: " + str(len(self._rl_vehicleID)))
  
  
		# if self._isTestFlag==False:
		# 	if ratioOfHaltVehicle >0.90:
		# 		print("Reset the Episode")
		# 		for agent in self.agents:
		# 			agent.done= True
		
		self.agents = self.createNDynamicAgents()

		for agent in self.agents:
			agent.done = False

		for agent in self.agents:
			obs_n.append(self._get_obs(agent))	
			# print(self._get_obs(agent))		
			reward_n.append(self._get_reward(agent))
			# print(self._get_reward(agent))
			done_n.append(self._get_done(agent))

			info_n['n'].append(self._get_info(agent))

		if self._reward_type == "Global":
			self._currentReward = reward_n
			reward = np.sum(reward_n)/len(self.agents)
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
		
		while len(obs_n)<self.n:
			obs_n.append([0,0,0,0,0,0,0])
			done_n.append(False)
			info_n['n'].append({})


		return obs_n, newReward_n, done_n, info_n

	# set env action for a particular agent
	def _set_action(self,time=None):
		# process action
		#index 0 = # set default
		#index 1 = # set priority
		if self.SotaFlag==True:
			self.setRLAgentTogglePriority() # to simulate human decision-making
			# print("Inside Sota")
			return
		if self.scenario_flag not in ['baseline1','baseline2']:	
			for agent in self.agents: #loop through all agent
				agent_id = f'{agent.id}'
				action = self.lastActionDict[agent_id]
				if action == 0:
					if self.traci.vehicle.getTypeID(agent_id)!="rl-default":
						self._collisionCounter+=1
						self.traci.vehicle.setType(agent_id,"rl-default")
					# lane_index = self.traci.vehicle.getLaneIndex(agent_id)
					# if lane_index==0:
					# 	self.traci.vehicle.changeLane(agent_id,1,0)
				elif action == 1:
					if self.traci.vehicle.getTypeID(agent_id)!="rl-priority":
						self._collisionCounter+=1
						self.traci.vehicle.setType(agent_id,"rl-priority")
					# lane_index = self.traci.vehicle.getLaneIndex(agent_id)
					# if lane_index!=0:
						# self.traci.vehicle.changeLane(agent_id,0,0)
	
					
	
	def initSimulator(self,withGUI,portnum):
		if withGUI:
			import traci
		else:
			try:
				import libsumo as traci
			except:
				import traci
		seed = self._sumo_seed
  
		self.sumoCMD = ["-c", self.sumoConfig, "-r", self._routeFileName, "--waiting-time-memory",str(self.waiting_time_memory),"--scale",str(self.scaleFactor), "-W"]
		# Define experiment-specific args here
		if self._isTestFlag==True:
			self.sumoCMD += ["--lanechange-output",f"laneChange_stats_{seed}.xml","--edgedata-output",f"edge_stats_{seed}.xml"] # for metric logging
			self.sumoCMD += ["--time-to-teleport.disconnected", str(10), "--collision.action","teleport"] # prevents simulation from clogging up
		else:				
			self.sumoCMD += ["--time-to-teleport", str(-1), "--collision.action","none"] # prevents RL agents from vanishing during training
		# "-W", "--default.carfollowmodel", "IDM","--max-num-vehicles",str(300),"--tripinfo","tripOutput.xml",
		if withGUI:
			sumoBinary = checkBinary('sumo-gui')
			self.sumoCMD += ["--start", "--quit-on-end"]
		else:	
			sumoBinary = checkBinary('sumo')
# "--ignore-route-errors",
		# print(sumoBinary)
		# sumoConfig = "sumo_configs/sim.sumocfg"
# "--lanechange.duration",str(2),

		print(' '.join([sumoBinary] + ["--seed", str(seed)] + self.sumoCMD))
		traci.start([sumoBinary] + ["--seed", str(seed)] + self.sumoCMD)

		self.lane2tls = {}
		for tls_id in traci.trafficlight.getIDList():
			for signal in traci.trafficlight.getControlledLinks(tls_id):
				for inlane, outlane, _ in signal: # connections
					self.lane2tls[inlane] = tls_id

		return traci

	def close(self):
		# print("RL_Counter :" + str(self._rl_counter))
		# print("CAV Flow " + str(self._cavFlowCounter))
		self.traci.close()
		sys.stdout.flush()

