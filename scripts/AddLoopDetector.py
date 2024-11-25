import numpy as np
import os, sys
from sumolib import net

from lxml import etree as ET
import csv


def loopDetector(network,edge_list,filename):
    
    dataList = []
    for edge_id in edge_list:
        lanes = network.getEdge(edge_id).getLanes()

        for lane in lanes:
            length = lane.getLength()
            lane_id = lane.getID()
            lane_number = lane_id.split("_")[1]
            loopDetectorId = "det_" + lane_id + "_" + str(1)
            pos = "-1"
            data = [lane_id,loopDetectorId,pos]
            dataList.append(data)

    # write dataList to a CSV file
    # header = ['Lane_ID','Loop Detector Id', 'Loop Detector Position']
    with open('./sumo_configs/Test/Barcelona/loopDetectorList.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        # writer.writerow(header)
        # write multiple rows
        writer.writerows(dataList)

    # write additional file for sumocfg
    writeAdditionalFilesForLoopDetector(edge_list)

vehicleTypeMap = {'passenger': 'custom2 custom1 passenger'}

def writeAdditionalFilesForLoopDetector(edge_list):
   
    data = ET.Element('additionals')
    with open('./sumo_configs/Test/Barcelona/loopDetectorList.csv', 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            for key, vtype in vehicleTypeMap.items():
                s_elem1 = ET.SubElement(data, 'inductionLoop')
                s_elem1.set('id', f'{row[1]}_{key}')
                s_elem1.set('lane', row[0])
                s_elem1.set('pos', row[2])
                s_elem1.set('freq', str(300))
                # s_elem1.set('vTypes', vtype)
                s_elem1.set('file', 'loopDetectors.out.xml')

    b_xml = ET.tostring(data, pretty_print=True)
 
    # Opening a file under the name `items2.xml`,
    # with operation mode `wb` (write + binary)
    with open("./sumo_configs/Test/Barcelona/Barcelona_loopDetectors.add.xml", "wb") as f:
        f.write(b_xml)


if __name__=="__main__":
    loopDetectorFileName = "./sumo_configs/Test/Barcelona/Barcelona_loopDetectors.add.xml"
    networkFileName = './sumo_configs/Test/Barcelona/EIXAMPLE02_baseline_000.net.xml'


    step = 0

    network = net.readNet(networkFileName)

    edge_list = [e.getID() for e in network.getEdges(withInternal=False)] # list of all edges excluding internal links

    # Write additional file with loop detectors if the file does not exist

    ###uncomment below function everytime you need to generate new Loop  detector additional file
    loopDetector(network, edge_list, loopDetectorFileName)
    ###uncomment below function everytime you need to generate new Loop  detector additional file

