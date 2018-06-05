import atexit
from queue import Queue
from threading import Event
from threading import Thread

from lxml import etree
from mujoco_py import load_model_from_xml
import numpy as np

from rllab.dynamics_randomization import Distribution
from rllab.dynamics_randomization import Method


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
        # Synchronized queue to store mujoco_models
        self._models = Queue(maxsize=10)
        # Worker Thread
        self._worker_thread = Thread(
            target=self._generator_routine, daemon=True, name="Worker-Thread")
        # Reference to the generated model
        self._mujoco_model = None
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
            # If worker thread terminates because of an error, terminates main thread
            raise ChildProcessError("Error raised in Worker-Thread")

        return self._models.get()

    def stop(self):
        """
        Stops the worker thread. This method has to be called when the corresponding
        randomized environment is terminated or when the training is interrupted.
        """
        if self._worker_thread.is_alive():
            while not self._models.empty():
                self._models.get()

            self._stop_event.set()
            self._worker_thread.join()

    def _generator_routine(self):
        """
        Routine of the worker thread in this class.
        """
        # Initialize parsing of the model from XML
        parsed_model = etree.parse(self._file_path)
        elem_cache = {}
        default_cache = {}
        for v in self._variations.get_list():
            e = parsed_model.find(v.xpath)
            if e is None:
                raise ValueError(
                    "Could not find node in the XML model: %s" % v.xpath)
            elem_cache[v] = e

            if v.attrib not in e.attrib:
                raise ValueError("Attribute %s doesn't exist in node %s" %
                                 (v.attrib, v.xpath))
            val = e.attrib[v.attrib].split(' ')
            if len(val) == 1:
                default_cache[v] = float(e.attrib[v.attrib])
            else:
                default_cache[v] = np.array(list(map(float, val)))

            if len(v.var_range) != 2 * len(val):
                raise ValueError("Range shape != default value shape")

        # Generate model with randomized dynamic parameters
        while not self._stop_event.is_set():
            for v in self._variations.get_list():
                e = elem_cache[v]
                if v.distribution == Distribution.GAUSSIAN:
                    c = np.random.normal(
                        loc=v.mean_std[0], scale=v.mean_std[1])
                elif v.distribution == Distribution.UNIFORM:
                    c = np.random.uniform(
                        low=v.var_range[0], high=v.var_range[1])
                else:
                    raise ValueError("Unknown distribution")
                if v.method == Method.COEFFICIENT:
                    e.attrib[v.attrib] = str(c * default_cache[v])
                elif v.method == Method.ABSOLUTE:
                    e.attrib[v.attrib] = str(c)
                else:
                    raise ValueError("Unknown method")

            model_xml = etree.tostring(parsed_model.getroot()).decode("ascii")
            self._mujoco_model = load_model_from_xml(model_xml)
            self._models.put(self._mujoco_model)
