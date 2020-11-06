"""Tools for running experiments with Garage."""
import collections
import datetime
import enum
import functools
import gc
import inspect
import json
import os
import pathlib
import subprocess
import warnings
import weakref

import dateutil.tz
import dowel
from dowel import logger
import numpy as np

import __main__ as main

exp_count = 0
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
EIGHT_MEBIBYTES = 8 * 2**20


def _make_sequential_log_dir(log_dir):
    """Creates log_dir, appending a number if necessary.

    Attempts to create the directory `log_dir`. If it already exists, appends
    "_1". If that already exists, appends "_2" instead, etc.

    Args:
        log_dir (str): The log directory to attempt to create.

    Returns:
        str: The log directory actually created.

    """
    i = 0
    while True:
        try:
            if i == 0:
                os.makedirs(log_dir)
            else:
                possible_log_dir = '{}_{}'.format(log_dir, i)
                os.makedirs(possible_log_dir)
                log_dir = possible_log_dir
            return log_dir
        except FileExistsError:
            i += 1


def _make_experiment_signature(function):
    """Generate an ExperimentTemplate's signature from its function.

    Checks that the first parameter is named ctxt and removes it from the
    signature. Makes all other parameters keyword only.

    Args:
        function (callable[ExperimentContext, ...]): The wrapped function.

    Returns:
        inspect.Signature: The signature of the ExperimentTemplate.

    Raises:
        ValueError: If the wrapped function's first parameter is not 'ctxt'.

    """
    func_sig = inspect.signature(function)
    new_params = []
    saw_first_param = False
    for param in func_sig.parameters.values():
        if not saw_first_param:
            # Don't output it to the experiment params, since it will contain
            # the context.
            if param.name != 'ctxt':
                raise ValueError(
                    'Experiment functions should have a first '
                    "parameter named 'ctxt' instead of {!r}".format(
                        param.name))
            saw_first_param = True
        else:
            new_params.append(
                inspect.Parameter(name=param.name,
                                  kind=inspect.Parameter.KEYWORD_ONLY,
                                  default=param.default,
                                  annotation=param.annotation))
    if not saw_first_param:
        raise ValueError(
            'Experiment functions should have a first parameter '
            "named 'ctxt', but {!r} has no parameters".format(function))
    return inspect.Signature(new_params,
                             return_annotation=func_sig.return_annotation)


class ExperimentContext:
    """Context in which an experiment is being run.

    Currently, this class implements the same interface as SnapshotConfig, but
    it will be extended in the future.

    Args:
        snapshot_dir (str): The full directory to put snapshots in.
        snapshot_mode (str): Policy for which snapshots to keep (or make at
            all). Can be either "all" (all iterations will be saved), "last"
            (only the last iteration will be saved), "gap" (every snapshot_gap
            iterations are saved), or "none" (do not save snapshots).
        snapshot_gap (int): Gap between snapshot iterations. Waits this number
            of iterations before taking another snapshot.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *, snapshot_dir, snapshot_mode, snapshot_gap):
        self.snapshot_dir = snapshot_dir
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap


class ExperimentTemplate:
    """Creates experiment log directories and runs an experiment.

    This class should only be created by calling garage.wrap_experiment.
    Generally, it's used as a decorator like this:

        @wrap_experiment(snapshot_mode='all')
        def my_experiment(ctxt, seed, lr=0.5):
            ...

        my_experiment(seed=1)

    Even though this class could be implemented as a closure in
    wrap_experiment(), it's more readable (and easier to pickle) implemented as
    a class.

    Note that the full path that will be created is
    f'{data}/local/{prefix}/{name}'.

    Args:
        function (callable or None): The experiment function to wrap.
        log_dir (str or None): The full log directory to log to. Will be
            computed from `name` if omitted.
        name (str or None): The name of this experiment template. Will be
            filled from the wrapped function's name if omitted.
        prefix (str): Directory under data/local in which to place the
            experiment directory.
        snapshot_mode (str): Policy for which snapshots to keep (or make at
            all). Can be either "all" (all iterations will be saved), "last"
            (only the last iteration will be saved), "gap" (every snapshot_gap
            iterations are saved), or "none" (do not save snapshots).
        snapshot_gap (int): Gap between snapshot iterations. Waits this number
            of iterations before taking another snapshot.
        archive_launch_repo (bool): Whether to save an archive of the
            repository containing the launcher script. This is a potentially
            expensive operation which is useful for ensuring reproducibility.
        name_parameters (str or None): Parameters to insert into the experiment
            name. Should be either None (the default), 'all' (all parameters
            will be used), or 'passed' (only passed parameters will be used).
            The used parameters will be inserted in the order they appear in
            the function definition.
        use_existing_dir (bool): If true, (re)use the directory for this
            experiment, even if it already contains data.
        x_axis (str): Key to use for x axis of plots.



    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *, function, log_dir, name, prefix, snapshot_mode,
                 snapshot_gap, archive_launch_repo, name_parameters,
                 use_existing_dir, x_axis):
        self.function = function
        self.log_dir = log_dir
        self.name = name
        self.prefix = prefix
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap
        self.archive_launch_repo = archive_launch_repo
        self.name_parameters = name_parameters
        self.use_existing_dir = use_existing_dir
        self.x_axis = x_axis
        if self.function is not None:
            self._update_wrap_params()

    def _update_wrap_params(self):
        """Update self to "look like" the wrapped funciton.

        Mostly, this involves creating a function signature for the
        ExperimentTemplate that looks like the wrapped function, but with the
        first argument (ctxt) excluded, and all other arguments required to be
        keyword only.

        """
        functools.update_wrapper(self, self.function)
        self.__signature__ = _make_experiment_signature(self.function)

    @classmethod
    def _augment_name(cls, options, name, params):
        """Augment the experiment name with parameters.

        Args:
            options (dict): Options to `wrap_experiment` itself. See the
                function documentation for details.
            name (str): Name without parameter names.
            params (dict): Dictionary of parameters.

        Raises:
            ValueError: If self.name_parameters is not set to None, "passed",
                or "all".

        Returns:
            str: Returns the augmented name.

        """
        name_parameters = collections.OrderedDict()

        if options['name_parameters'] == 'passed':
            for param in options['signature'].parameters.values():
                try:
                    name_parameters[param.name] = params[param.name]
                except KeyError:
                    pass
        elif options['name_parameters'] == 'all':
            for param in options['signature'].parameters.values():
                name_parameters[param.name] = params.get(
                    param.name, param.default)
        elif options['name_parameters'] is not None:
            raise ValueError('wrap_experiment.name_parameters should be set '
                             'to one of None, "passed", or "all"')
        param_str = '_'.join('{}={}'.format(k, v)
                             for (k, v) in name_parameters.items())
        if param_str:
            return '{}_{}'.format(name, param_str)
        else:
            return name

    def _get_options(self, *args):
        """Get the options for wrap_experiment.

        This method combines options passed to `wrap_experiment` itself and to
        the wrapped experiment.

        Args:
            args (list[dict]): Unnamed arguments to the wrapped experiment. May
                be an empty list or a list containing a single dictionary.

        Raises:
            ValueError: If args contains more than one value, or the value is
                not a dictionary containing at most the same keys as are
                arguments to `wrap_experiment`.

        Returns:
            dict: The final options.

        """
        options = dict(name=self.name,
                       function=self.function,
                       prefix=self.prefix,
                       name_parameters=self.name_parameters,
                       log_dir=self.log_dir,
                       archive_launch_repo=self.archive_launch_repo,
                       snapshot_gap=self.snapshot_gap,
                       snapshot_mode=self.snapshot_mode,
                       use_existing_dir=self.use_existing_dir,
                       x_axis=self.x_axis,
                       signature=self.__signature__)
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                for k in args[0]:
                    if k not in options:
                        raise ValueError('Unknown key {} in wrap_experiment '
                                         'options'.format(k))
                options.update(args[0])
            else:
                raise ValueError('garage.experiment currently only supports '
                                 'keyword arguments')
        return options

    @classmethod
    def _make_context(cls, options, **kwargs):
        """Make a context from the template information and variant args.

        Currently, all arguments should be keyword arguments.

        Args:
            options (dict): Options to `wrap_experiment` itself. See the
                function documentation for details.
            kwargs (dict): Keyword arguments for the wrapped function. Will be
                logged to `variant.json`

        Returns:
            ExperimentContext: The created experiment context.

        """
        name = options['name']
        if name is None:
            name = options['function'].__name__
        name = cls._augment_name(options, name, kwargs)
        log_dir = options['log_dir']
        if log_dir is None:
            log_dir = ('{data}/local/{prefix}/{name}'.format(
                data=os.path.join(os.getcwd(), 'data'),
                prefix=options['prefix'],
                name=name))
        if options['use_existing_dir']:
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = _make_sequential_log_dir(log_dir)

        tabular_log_file = os.path.join(log_dir, 'progress.csv')
        text_log_file = os.path.join(log_dir, 'debug.log')
        variant_log_file = os.path.join(log_dir, 'variant.json')
        metadata_log_file = os.path.join(log_dir, 'metadata.json')

        dump_json(variant_log_file, kwargs)
        git_root_path, metadata = get_metadata()
        dump_json(metadata_log_file, metadata)
        if git_root_path and options['archive_launch_repo']:
            make_launcher_archive(git_root_path=git_root_path, log_dir=log_dir)

        logger.add_output(dowel.TextOutput(text_log_file))
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(
            dowel.TensorBoardOutput(log_dir, x_axis=options['x_axis']))
        logger.add_output(dowel.StdOutput())

        logger.push_prefix('[{}] '.format(name))
        logger.log('Logging to {}'.format(log_dir))

        return ExperimentContext(snapshot_dir=log_dir,
                                 snapshot_mode=options['snapshot_mode'],
                                 snapshot_gap=options['snapshot_gap'])

    def __call__(self, *args, **kwargs):
        """Wrap a function to turn it into an ExperimentTemplate.

        Note that this docstring will be overriden to match the function's
        docstring on the ExperimentTemplate once a function is passed in.

        Args:
            args (list): If no function has been set yet, must be a list
                containing a single callable. If the function has been set, may
                be a single value, a dictionary containing overrides for the
                original arguments to `wrap_experiment`.
            kwargs (dict): Arguments passed onto the wrapped function.

        Returns:
            object: The returned value of the wrapped function.

        Raises:
            ValueError: If not passed a single callable argument.

        """
        if self.function is None:
            if len(args) != 1 or len(kwargs) != 0 or not callable(args[0]):
                raise ValueError('Please apply the result of '
                                 'wrap_experiment() to a single function')
            # Apply ourselves as a decorator
            self.function = args[0]
            self._update_wrap_params()
            return self
        else:
            ctxt = self._make_context(self._get_options(*args), **kwargs)
            result = self.function(ctxt, **kwargs)
            logger.remove_all()
            logger.pop_prefix()
            gc.collect()  # See dowel issue #44
            return result


def wrap_experiment(function=None,
                    *,
                    log_dir=None,
                    prefix='experiment',
                    name=None,
                    snapshot_mode='last',
                    snapshot_gap=1,
                    archive_launch_repo=True,
                    name_parameters=None,
                    use_existing_dir=False,
                    x_axis='TotalEnvSteps'):
    """Decorate a function to turn it into an ExperimentTemplate.

    When invoked, the wrapped function will receive an ExperimentContext, which
    will contain the log directory into which the experiment should log
    information.

    This decorator can be invoked in two differed ways.

    Without arguments, like this:

        @wrap_experiment
        def my_experiment(ctxt, seed, lr=0.5):
            ...

    Or with arguments:

        @wrap_experiment(snapshot_mode='all')
        def my_experiment(ctxt, seed, lr=0.5):
            ...

    All arguments must be keyword arguments.

    Args:
        function (callable or None): The experiment function to wrap.
        log_dir (str or None): The full log directory to log to. Will be
            computed from `name` if omitted.
        name (str or None): The name of this experiment template. Will be
            filled from the wrapped function's name if omitted.
        prefix (str): Directory under data/local in which to place the
            experiment directory.
        snapshot_mode (str): Policy for which snapshots to keep (or make at
            all). Can be either "all" (all iterations will be saved), "last"
            (only the last iteration will be saved), "gap" (every snapshot_gap
            iterations are saved), or "none" (do not save snapshots).
        snapshot_gap (int): Gap between snapshot iterations. Waits this number
            of iterations before taking another snapshot.
        archive_launch_repo (bool): Whether to save an archive of the
            repository containing the launcher script. This is a potentially
            expensive operation which is useful for ensuring reproducibility.
        name_parameters (str or None): Parameters to insert into the experiment
            name. Should be either None (the default), 'all' (all parameters
            will be used), or 'passed' (only passed parameters will be used).
            The used parameters will be inserted in the order they appear in
            the function definition.
        use_existing_dir (bool): If true, (re)use the directory for this
            experiment, even if it already contains data.
        x_axis (str): Key to use for x axis of plots.

    Returns:
        callable: The wrapped function.

    """
    return ExperimentTemplate(function=function,
                              log_dir=log_dir,
                              prefix=prefix,
                              name=name,
                              snapshot_mode=snapshot_mode,
                              snapshot_gap=snapshot_gap,
                              archive_launch_repo=archive_launch_repo,
                              name_parameters=name_parameters,
                              use_existing_dir=use_existing_dir,
                              x_axis=x_axis)


def dump_json(filename, data):
    """Dump a dictionary to a file in JSON format.

    Args:
        filename(str): Filename for the file.
        data(dict): Data to save to file.

    """
    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        # We do our own circular reference handling.
        # Sometimes sort_keys fails because the keys don't get made into
        # strings early enough.
        json.dump(data,
                  f,
                  indent=2,
                  sort_keys=False,
                  cls=LogEncoder,
                  check_circular=False)


def get_metadata():
    """Get metadata about the main script.

    The goal of this function is to capture the additional information needed
    to re-run an experiment, assuming that the launcher script that started the
    experiment is located in a clean git repository.

    Returns:
        tuple[str, dict[str, str]]:
          * Absolute path to root directory of launcher's git repo.
          * Directory containing:
            * githash (str): Hash of the git revision of the repo the
                experiment was started from. "-dirty" will be appended to this
                string if the repo has uncommitted changes. May not be present
                if the main script is not in a git repo.
            * launcher (str): Relative path to the main script from the base of
                the repo the experiment was started from. If the main script
                was not started from a git repo, this will instead be an
                absolute path to the main script.

    """
    main_file = getattr(main, '__file__', None)
    if not main_file:
        return None, {}
    main_file_path = os.path.abspath(main_file)
    try:
        git_root_path = subprocess.check_output(
            ('git', 'rev-parse', '--show-toplevel'),
            cwd=os.path.dirname(main_file_path),
            stderr=subprocess.DEVNULL)
        git_root_path = git_root_path.strip()
    except subprocess.CalledProcessError:
        # This file is always considered not to exist.
        git_root_path = ''
    # We check that the path exists since in old versions of git the above
    # rev-parse command silently exits with 0 when run outside of a git repo.
    if not os.path.exists(git_root_path):
        return None, {
            'launcher': main_file_path,
        }
    launcher_path = os.path.relpath(bytes(main_file_path, encoding='utf8'),
                                    git_root_path)
    git_hash = subprocess.check_output(('git', 'rev-parse', 'HEAD'),
                                       cwd=git_root_path)
    git_hash = git_hash.decode('utf-8').strip()
    git_status = subprocess.check_output(('git', 'status', '--short'),
                                         cwd=git_root_path)
    git_status = git_status.decode('utf-8').strip()
    if git_status != '':
        git_hash = git_hash + '-dirty'
    return git_root_path, {
        'githash': git_hash,
        'launcher': launcher_path.decode('utf-8'),
    }


def make_launcher_archive(*, git_root_path, log_dir):
    """Saves an archive of the launcher's git repo to the log directory.

    Args:
        git_root_path (str): Absolute path to git repo to archive.
        log_dir (str): Absolute path to the log directory.

    """
    git_files = subprocess.check_output(
        ('git', 'ls-files', '--others', '--exclude-standard', '--cached',
         '-z'),
        cwd=git_root_path).strip()
    repo_size = 0
    files_to_archive = []
    for f in git_files.split(b'\0'):
        try:
            file_size = os.stat(os.path.join(git_root_path, f)).st_size
            repo_size += file_size
            if file_size < EIGHT_MEBIBYTES:
                files_to_archive.append(f)
        except FileNotFoundError:
            pass
    if repo_size >= EIGHT_MEBIBYTES:
        warnings.warn('Archiving a launch repo larger than 8MiB. This may be '
                      'slow. Set archive_launch_repo=False in wrap_experiment '
                      'to disable this behavior.')
    archive_path = os.path.join(log_dir, 'launch_archive.tar.xz')
    subprocess.run(('tar', '--null', '--files-from', '-', '--xz', '--create',
                    '--file', archive_path),
                   input=b'\0'.join(files_to_archive),
                   cwd=git_root_path,
                   check=True)


class LogEncoder(json.JSONEncoder):
    """Encoder to be used as cls in json.dump.

    Args:
        args (object): Passed to super class.
        kwargs (dict): Passed to super class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._markers = {}

    # Modules whose contents cannot be meaningfully or safelly jsonified.
    BLOCKED_MODULES = {
        'tensorflow',
        'ray',
        'itertools',
    }

    def default(self, o):
        """Perform JSON encoding.

        Args:
            o (object): Object to encode.

        Raises:
            TypeError: If `o` cannot be turned into JSON even using `repr(o)`.

        Returns:
            dict or str or float or bool: Object encoded in JSON.

        """
        # Why is this method hidden? What does that mean?
        # pylint: disable=method-hidden
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements
        # This circular reference checking code was copied from the standard
        # library json implementation, but it outputs a repr'd string instead
        # of ValueError on a circular reference.
        if isinstance(o, (int, bool, float, str)):
            return o
        else:
            markerid = id(o)
            if markerid in self._markers:
                return 'circular ' + repr(o)
            else:
                self._markers[markerid] = o
                try:
                    return self._default_inner(o)
                finally:
                    del self._markers[markerid]

    def _default_inner(self, o):
        """Perform JSON encoding.

        Args:
            o (object): Object to encode.

        Raises:
            TypeError: If `o` cannot be turned into JSON even using `repr(o)`.
            ValueError: If raised by calling repr on an object.

        Returns:
            dict or str or float or bool: Object encoded in JSON.

        """
        # Why is this method hidden? What does that mean?
        # pylint: disable=method-hidden
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements
        # This circular reference checking code was copied from the standard
        # library json implementation, but it outputs a repr'd string instead
        # of ValueError on a circular reference.
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError as err:
            if isinstance(o, dict):
                data = {}
                for (k, v) in o.items():
                    if isinstance(k, str):
                        data[k] = self.default(v)
                    else:
                        data[repr(k)] = self.default(v)
                return data
            elif isinstance(o, weakref.ref):
                return repr(o)
            elif type(o).__module__.split('.')[0] in self.BLOCKED_MODULES:
                return repr(o)
            elif isinstance(o, type):
                return {'$typename': o.__module__ + '.' + o.__name__}
            elif isinstance(o, np.number):
                # For some reason these aren't natively considered
                # serializable.
                # JSON doesn't actually have ints, so always use a float.
                return float(o)
            elif isinstance(o, np.bool8):
                return bool(o)
            elif isinstance(o, enum.Enum):
                return {
                    '$enum':
                    o.__module__ + '.' + o.__class__.__name__ + '.' + o.name
                }
            elif isinstance(o, np.ndarray):
                return repr(o)
            elif hasattr(o, '__dict__') or hasattr(o, '__slots__'):
                obj_dict = getattr(o, '__dict__', None)
                if obj_dict is not None:
                    data = {k: self.default(v) for (k, v) in obj_dict.items()}
                else:
                    data = {
                        s: self.default(getattr(o, s))
                        for s in o.__slots__
                    }
                t = type(o)
                data['$type'] = t.__module__ + '.' + t.__name__
                return data
            elif callable(o) and hasattr(o, '__name__'):
                if getattr(o, '__module__', None) is not None:
                    return {'$function': o.__module__ + '.' + o.__name__}
                else:
                    return repr(o)
            else:
                try:
                    # This case handles many built-in datatypes like deques
                    return [self.default(v) for v in list(o)]
                except TypeError:
                    pass
                try:
                    # This case handles most other weird objects.
                    return repr(o)
                except TypeError:
                    pass
                raise err
