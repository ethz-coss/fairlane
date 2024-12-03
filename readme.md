# FAIRLANE
This repository hosts the code for the paper *"FAIRLANE: A multi-agent approach to priority lane management in diverse traffic composition"*.

## Citing
If you found any part of this repository useful for your work, please cite our paper:
```bibtex
@article{dubery2024fairlane,
  title = {{FAIRLANE}: A multi-agent approach to priority lane management in diverse traffic composition},
  volume = {171},
  DOI = {10.1016/j.trc.2024.104919},
  journal = {Transportation Research Part C: Emerging Technologies},
  publisher = {Elsevier BV},
  author = {Dubey,  Rohit K. and Dailisan,  Damian and Argota Sánchez–Vaquerizo,  Javier  and Helbing,  Dirk},
  year = {2025},
  pages = {104919}
}
```

## Running


1. Create a python virtual env
```bash
python -m venv sumo
```
Note that **sumo** here can be replaced with any venv name of your choosing.

2. Activate the venv
```bash
source activate sumo/bin/activate
```

3. Install packages
```bash
pip install -r requirements.txt
mkdir results
```

4. Define `SUMO_HOME` to your bashrc file. If you are using a different shell, edit this accordingly. If you choose to install sumo via other means, also change the `SUMO_HOME` location.
```
echo  'export SUMO_HOME=$(python -m site --user-site)/sumo' >> ~/.bashrc
export SUMO_HOME=$(python -m site --user-site)/sumo
```

5. Train the model using 
```bash
python train.py # trains the MADDPG model
python run_mappo.py --option train # trains the MAPPO model
```

6. Run all of the ablation tests using 
```bash
python run_tests.py
```
*NOTE:* You must repeat step 2 when running code. 


