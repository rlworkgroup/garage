#!/usr/bin/env python3
"""
Benchmark model mutation for dynamics randomization
"""
from mujoco_py import load_model_from_xml
from mujoco_py import MjSim
from mujoco_py import MjViewer
from rllab.dynamics_randomization import Variations, VariationMethod
from rllab.dynamics_randomization import VariationDistribution

import numpy as np
import os
import os.path as osp
import xml.etree.ElementTree as ET

#Execute at the root of rllab
MUJOCO_PY_PATH = os.getcwd()
TOSSER_XML = osp.join(MUJOCO_PY_PATH,
                      "rllab/dynamics_randomization/tosser.xml")

# Load original model text into memory
tosser = ET.parse(TOSSER_XML)

variations = Variations()
variations.randomize().\
        attribute("gear").\
        at_xpath(".//motor[@name='a1']").\
        with_method(VariationMethod.COEFFICIENT).\
        sampled_from(VariationDistribution.UNIFORM).\
        with_range(0.5, 1.5).\
        randomize().\
        attribute("gear").\
        at_xpath(".//motor[@name='a2']").\
        sampled_from(VariationDistribution.UNIFORM).\
        with_method(VariationMethod.COEFFICIENT).\
        with_range(0.5, 1.5)

variations.randomize().\
        attribute("damping").\
        at_xpath(".//joint[@name='wr_js']").\
        with_method(VariationMethod.ABSOLUTE).\
        sampled_from(VariationDistribution.UNIFORM).\
        with_range(5, 15)

# Retrieve defaults and cache etree elems
for v in variations.get_list():
    e = tosser.find(v.xpath)
    v.elem = e
    v.default = float(e.attrib[v.attrib])

for _ in range(1000):
    # Mutate model randomly
    for v in variations.get_list():
        e = v.elem
        if v.method == VariationMethod.COEFFICIENT:
            c = np.random.uniform(low=v.var_range[0], high=v.var_range[1])
            e.attrib[v.attrib] = str(c * v.default)
        elif v.method == VariationMethod.ABSOLUTE:
            c = np.random.uniform(low=v.var_range[0], high=v.var_range[1])
            e.attrib[v.attrib] = str(c)
        else:
            raise NotImplementedError("Unknown method")

    # Reify model
    model_xml = ET.tostring(tosser.getroot()).decode("ascii")

    # Run model loop
    model = load_model_from_xml(model_xml)
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
