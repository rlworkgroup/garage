import threading.Thread
import threading.RLock
import threading.Event
import queue.Queue

'''
A worker thread to produce to MuJoCo models with randomized dynamic
parameters, which are specified by the users of rllab with the class
Variations.
'''
class MujocoModelGenerator:
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
    def __init__(self, file_path, variations):
        self._parsed_model = etree.parse(file_path)
        self._variations = variations 

        for v in variations.get_list():
            e = self.parsed_model.find(v.xpath)
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
        # Worker Thread   
        self._worker_thread = Thread(target=self._generator_routine)
        # Reference to the generated model
        self._mujoco_model = None
        # Communicates the calling thread with the worker thread by awaking 
        # the worker thread so as to generate a new model.
        self._model_requested = Event()
        # Communicates the worker thread with the calling thread by awaking 
        # the calling thread so as to retrieve the generated model.
        self._model_ready = Event()
        self._worker_thread.start()


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
    def get_model(self):
        if not self._model_ready.is_set():
            # If the model is not ready yet, wait for it to be finished.
            self._model_ready.wait()
        # Cleat the event flag for the next iteration
        self._model_ready.clear()
        # Request a new model to the worker thread.
        self._model_requested.set()
        return self._mujoco_model

    def _generator_routine(self):
        while(True):
            for v in self._variations.get_list():
                e = v.elem
                if v.distribution == VariationDistributions.GAUSSIAN:
                    c = np.random.normal(loc=v.var_range[0], scale=v.var_range[1])
                elif v.distribution == VariationDistributions.UNIFORM:
                    c = np.random.uniform(low=v.var_range[0], high=v.var_range[1])
                else:
                    raise NotImplementedError("Unkown distribution")
                if v.method == VariationMethods.COEFFICIENT:
                    e.attrib[v.attrib] = str(c * v.default)
                elif v.method == VariationMethods.ABSOLUTE:
                    e.attrib[v.attrib] = str(c)
                else:
                    raise NotImplementedError("Unknown method")

            model_xml = etree.tostring(self._parsed_model.getroot()).decode("ascii")
            self._mujoco_model = load_model_from_xml(model_xml)

            # Wake up the calling thread if it was waiting
            self._model_ready.set()
            # Go to idle mode (wait for event)
            self._model_requested.wait()
            self._model_requested.clear()
