from lxml import etree as ET
from sumolib import checkBinary
import os
import sys
import numpy as np
import traci
import csv
def loopDetector(network,edge_list,filename):
    
    dataList = []
    for edge_id in edge_list:
        lanes = network.getEdge(edge_id).getLanes()

        # lane_index = 0
        # if edge_id == "-840":
        for lane in lanes:
            # lane_id = edge_id + "_" + str(lane_index)
            # length = traci.lane.getLength(lane_id)
            length = lane.getLength()
            lane_id = lane.getID()
            lane_number = lane_id.split("_")[1]
            # # divide the lane into N segements of 25 meters
            # numberOfSegment = int(length/params.segmentLength)
            # loopCounter = 0
            # pos = 0
            # while loopCounter < numberOfSegment:
            #     loopDetectorId = "det_" + lane_id + "_" + str(loopCounter)
            #     pos = 12.5 + 25*loopCounter              
            #     print(lane_id,loopDetectorId,"---",pos)
            #     data = [lane_id,loopDetectorId,pos]
            #     dataList.append(data)
            #     loopCounter+=1

            # for scenario when two loop detectors are needed. One at the begining and another at the end.
            # loopDetectorId = "det_" + lane_id + "_" + str(0)
            # pos = 0
            # data = [lane_id,loopDetectorId,pos]
            # dataList.append(data)
            if lane_number=="0":
                loopDetectorId = "det_" + lane_id + "_" + str(1)
                pos = length - 5
                data = [lane_id,loopDetectorId,pos]
                dataList.append(data)

            
            # print(numberOfSegment)
            # lane_index+=1

    # write dataList to a CSV file
    # header = ['Lane_ID','Loop Detector Id', 'Loop Detector Position']
    with open('D:\Rohit\prioritylane/sumo_configs/loopDetectorList.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        # writer.writerow(header)
        # write multiple rows
        writer.writerows(dataList)
                # print(length)

    # write additional file for sumocfg
    writeAdditionalFilesForLoopDetector(edge_list)

vehicleTypeMap = {'passenger': 'custom2 custom1 passenger'}

def writeAdditionalFilesForLoopDetector(edge_list):
   
    data = ET.Element('additionals')
    with open('D:\Rohit\prioritylane/sumo_configs/loopDetectorList.csv', 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            for key, vtype in vehicleTypeMap.items():
                s_elem1 = ET.SubElement(data, 'e1Detector')
                s_elem1.set('id', f'{row[1]}_{key}')
                s_elem1.set('lane', row[0])
                s_elem1.set('pos', row[2])
                s_elem1.set('freq', str(900))
                s_elem1.set('vTypes', vtype)
                s_elem1.set('file', 'loopDetectors.out.xml')

    # edata_cars = ET.SubElement(data, 'edgeData')
    # edata_cars.set('id', 'edgestats_cars')
    # edata_cars.set('freq', str(600))
    # edata_cars.set('file', 'edgestats.out.xml')

    b_xml = ET.tostring(data, pretty_print=True)
 
    # Opening a file under the name `items2.xml`,
    # with operation mode `wb` (write + binary)
    with open("D:\Rohit\prioritylane/sumo_configs/LTN_loopDetectors.add.xml", "wb") as f:
        f.write(b_xml)


            





