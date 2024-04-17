"""
This script reads in the network given as
 first parameter and generates additional connections at intersections.
"""
from __future__ import absolute_import
from __future__ import print_function


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sumolib.net  # noqa
from sumolib import checkBinary

def writeConnections(net):
    fd = open("patch-connection.con.xml", "w")
    fd.write("<connections>\n")
    for node in net.getNodes():
        edgesOut = node._outgoing
        edgesIn = node._incoming
        for edgeOut in edgesOut:
            outNumber = edgeOut.getLaneNumber()
            for edgeIn in edgesIn:
                if edgeOut not in edgeIn._outgoing:
                    continue
                inNumber = edgeIn.getLaneNumber()
                ## check for turnarounds
                if edgeIn.getFromNode().getID()==edgeOut.getToNode().getID():
                    continue
                for x in range(inNumber):
                    if x < inNumber and x < outNumber:
                        fd.write(f'''\t<connection from="{edgeIn._id}" to="{edgeOut._id}" lane="{x}:{x}"/>\n''')
    fd.write("</connections>\n")


if __name__ == "__main__":
    import subprocess
    op = sumolib.options.ArgumentParser(
        description='Create connections in roundabout')

    op.add_argument("-n", "--net-file", category="input", type=op.net_file, dest="net", required=True,
                    help='Input file name')

    try:
        options = op.parse_args()
    except (NotImplementedError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print("Reading net...")
    net = sumolib.net.readNet(options.net)
    print("Writing connections...")
    writeConnections(net)
    netconvert = checkBinary('netconvert')
    netconvert_call = f"'{netconvert}' -s {options.net} -W -x patch-connection.con.xml -o patched.net.xml"
    print(netconvert_call)
    subprocess.run(netconvert_call, shell=True)