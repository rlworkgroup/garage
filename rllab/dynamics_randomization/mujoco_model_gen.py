import atexit
import sys
from threading import Event
from threading import Thread

from lxml import etree
from mujoco_py import load_model_from_xml
import numpy as np

from rllab.dynamics_randomization import VariationDistribution
from rllab.dynamics_randomization import VariationMethod


class MujocoModelGenerator:
    """
    A worker thread to produce to MuJoCo models with randomized dynamic
    parameters, which are specified by the users of rllab with the class
    Variations.
    """

    def __init__(self, file_path, variations):
        """
        Starts all the member fields of the class and the worker thread.
        Parameters
        ----------
        file_path : string
            The absolute path to the XML file that contains the MuJoCo
            model.
        variations: Variations
            An list of Variation objects that indicate the dynamic parameters
            to randomize in the XML file.
        """
        self._variations = variations
        self._file_path = file_path
        # Worker Thread
        self._worker_thread = Thread(
            target=self._generator_routine, daemon=True, name="Worker-Thread")
        # Reference to the generated model
        self._mujoco_model = None
        # Communicates the calling thread with the worker thread by awaking
        # the worker thread so as to generate a new model.
        self._model_requested = Event()
        # Communicates the worker thread with the calling thread by awaking
        # the calling thread so as to retrieve the generated model.
        self._model_ready = Event()
        # Event to stop the worker thread
        self._stop_event = Event()
        atexit.register(self.stop)
        self._worker_thread.start()

    def get_model(self):
        """
        Gets the MuJoCo model produced by the worker thread in this class.
        This call may block in case the calling thread asks for the model before
        the worker thread has finished.
        Returns
        -------
        PyMjModel
            A MuJoCo model with randomized dynamic parameters specified by the
            user in this class.
        """
        if not self._worker_thread.is_alive():
            # If worker thread is dead because of an error, raise an error in main thread
            raise ChildProcessError("Error raised in Worker-Thread")

        if not self._model_ready.is_set():
            # If the model is not ready yet, wait for it to be finished.
            self._model_ready.wait()
        # Cleat the event flag for the next iteration
        self._model_ready.clear()
        # Request a new model to the worker thread.
        self._model_requested.set()
        return self._mujoco_model

    def stop(self):
        """
        Stops the worker thread. This method has to be called when the corresponding
        randomized environment is terminated or when the training is interrupted.
        """
        if self._worker_thread.is_alive():
            self._model_requested.set()
            self._stop_event.set()
            self._worker_thread.join()

    def _generator_routine(self):
        """
        Routine of the worker thread in this class.
        """
        # Initialize parsing of the model from XML
        parsed_model = etree.parse(self._file_path)
        for v in self._variations.get_list():
            e = parsed_model.find(v.xpath)
            if e is None:
                raise AttributeError("Can't find node in xml")
            v.elem = e

            if v.attrib not in e.attrib:
                raise KeyError("Attribute doesn't exist")
            val = e.attrib[v.attrib].split(' ')
            if len(val) == 1:
                v.default = float(e.attrib[v.attrib])
            else:
                v.default = np.array(list(map(float, val)))

            if len(v.var_range) != 2 * len(val):
                raise AttributeError("Range shape != default value shape")

        # Generate model with randomized dynamic parameters
        while not self._stop_event.is_set():
            for v in self._variations.get_list():
                e = v.elem
                if v.distribution == VariationDistribution.GAUSSIAN:
                    c = np.random.normal(
                        loc=v.var_range[0], scale=v.var_range[1])
                elif v.distribution == VariationDistribution.UNIFORM:
                    c = np.random.uniform(
                        low=v.var_range[0], high=v.var_range[1])
                else:
                    raise NotImplementedError("Unknown distribution")
                if v.method == VariationMethod.COEFFICIENT:
                    e.attrib[v.attrib] = str(c * v.default)
                elif v.method == VariationMethod.ABSOLUTE:
                    e.attrib[v.attrib] = str(c)
                else:
                    raise NotImplementedError("Unknown method")

            model_xml = etree.tostring(parsed_model.getroot()).decode("ascii")
            self._mujoco_model = load_model_from_xml(model_xml)

            # Wake up the calling thread if it was waiting
            self._model_ready.set()
            # Go to idle mode (wait for event)
            self._model_requested.wait()
            self._model_requested.clear()
