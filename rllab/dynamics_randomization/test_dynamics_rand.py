#!/usr/bin/env python3
"""
Benchmark model mutation for dynamics randomization
"""
import os
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mujoco_py import load_model_from_xml
from mujoco_py import MjSim
from mujoco_py import MjViewer

from rllab.dynamics_randomization import VariationsList

#Execute at the root of rllab
MUJOCO_PY_PATH = os.getcwd()
TOSSER_XML = osp.join(MUJOCO_PY_PATH, "rllab/dynamics_randomization/tosser.xml")

# Load original model text into memory
tosser = ET.parse(TOSSER_XML)

var_list = VariationsList().\
    add_variation(".//motor[@name='a1']", "gear", "coefficient", "uniform", (0.5, 1.5)).\
    add_variation(".//motor[@name='a2']", "gear", "coefficient", "uniform", (0.5, 1.5)).\
    add_variation(".//joint[@name='wr_js']", "damping", "absolute", "uniform", (5, 15))

# Retrieve defaults and cache etree elems
for v in var_list.get_list():
    e = tosser.find(v.xpath)
    v.elem = e
    v.default = float(e.attrib[v.attrib])
    print(e)
    print(v.default)

for _ in range(1000):
    # Mutate model randomly
    for v in var_list.get_list():
        e = v.elem
        if v.method == "coefficient":
            c = np.random.uniform(low=v.var_range[0], high=v.var_range[1])
            e.attrib[v.attrib] = str(c * v.default)
        elif v.method == "absolute":
            c = np.random.uniform(low=v.var_range[0], high=v.var_range[1])
            e.attrib[v.attrib] = str(c)
        else:
            raise NotImplementedError("Unknown method")

    # Reify model
    model_xml = ET.tostring(tosser.getroot()).decode("ascii")

    # Run model loop
    model = load_model_from_xml(model_xml)
    print(model_xml)
    sim = MjSim(model)
    #viewer = MjViewer(sim)

    #sim_state = sim.get_state()


    #sim.set_state(sim_state)

    for i in range(1000):
        if i < 150:
            sim.data.ctrl[:] = 0.0
        else:
            sim.data.ctrl[:] = -1.0
        sim.step()
        #viewer.render()
