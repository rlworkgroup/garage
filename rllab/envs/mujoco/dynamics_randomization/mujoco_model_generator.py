import atexit
import queue
from queue import Queue
from threading import Event
from threading import Thread

from mujoco_py import load_model_from_xml


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

        try:
            return self._models.get(timeout=1)
        except queue.Empty:
            # If the queue is empty after 1s, there's something wrong in the worker thread
            raise ChildProcessError("Error raised in Worker-Thread")

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
        self._variations.initialize_variations(self._file_path)
        # Generate model with randomized dynamic parameters
        while not self._stop_event.is_set():
            self._mujoco_model = load_model_from_xml(
                self._variations.get_randomized_xml_model())
            self._models.put(self._mujoco_model)
