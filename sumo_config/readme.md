# Commands

This documents the commands needed to generate a synthetic network with arbitrarily generated car and bus flows.
The scripts in this repository build on `activitygen` of sumo.

We are assuming that you have installed sumo and have added it to your system `PATH`.
The `SUMO_HOME` environment variable should also be defined.

## Generating a random network

We can first generate a random network using `netgenerate`.
```
netgenerate --rand -o SimpleRandom.net.xml --rand.iterations=100 --default.lanenumber=3 --default.lanewidth=3.2 -j "traffic_light"
```


## Setting up activitygen

We provide a default template for activitygen in `activitygen_base.stat.xml`. It requires four inputs from the network files. 
1. `<streets>`
2. `<cityGates>` It is done manually for now. 
3. `<busStations>`
4. `<busLines>`

Step 2 has to be done manually by selecting candidate edges to be labeled as `cityGates`.
Bus lines are hardcoded in the file. Probably don't touch it. Routes.json file should exist along with the network. Few parameters
such as rate of bus line can be edited in the script. 

Running `scripts/main.py` should take care of generating 1, 3, and 4 are generated using the function `writeActivityGenSupportingData` in `scripts/writeActivityGenSupportingData.py` file.
Once this is run, all of the files loaded by `SimpleRandom.sumocfg` should be updated and ready for running a simulation.

```
sumo-gui -c SimpleRandom.sumocfg --start
```

Now, believe in God, and see if it works. 


duarouter -n SimpleRandom.net.xml --route-files SimpleRandom.trips.xml -o SimpleRandom.rou.xml --ignore-errors

duarouter -n SimpleRandom.net.xml --route-files SimpleRandom.bus.xml -o SimpleRandom_Bus.rou.xml --ignore-errors




