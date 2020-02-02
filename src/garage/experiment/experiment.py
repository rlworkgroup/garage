"""Tools for running experiments with Garage."""
# flake8: noqa
import base64
import datetime
import enum
import functools
import gc
import json
import os
import os.path as osp
import pathlib
import pickle
import re
import subprocess
import warnings

import cloudpickle
import dowel
from dowel import logger
import dateutil.tz
import __main__ as main

exp_count = 0
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
EIGHT_MEBIBYTES = 8 * 2**20


def run_experiment(method_call=None,
                   batch_tasks=None,
                   exp_prefix='experiment',
                   exp_name=None,
                   log_dir=None,
                   script='garage.experiment.experiment_wrapper',
                   python_command='python',
                   dry=False,
                   env=None,
                   variant=None,
                   force_cpu=False,
                   pre_commands=None,
                   **kwargs):
    """Serialize the method call and run the experiment using the
    specified mode.

    Args:
        method_call (callable): A method call.
        batch_tasks (list[dict]): A batch of method calls.
        exp_prefix (str): Name prefix for the experiment.
        exp_name (str): Name of the experiment.
        log_dir (str): Log directory for the experiment.
        script (str): The name of the entrance point python script.
        python_command (str): Python command to run the experiment.
        dry (bool): Whether to do a dry-run, which only prints the
            commands without executing them.
        env (dict): Extra environment variables.
        variant (dict): If provided, should be a dictionary of parameters.
        force_cpu (bool): Whether to set all GPU devices invisible
            to force use CPU.
        pre_commands (str): Pre commands to run the experiment.
        kwargs (dict): Additional parameters.

    """
    # pylint: disable=missing-raises-doc,global-statement,too-many-branches
    if method_call is None and batch_tasks is None:
        raise Exception(
            'Must provide at least either method_call or batch_tasks')

    for task in (batch_tasks or [method_call]):
        if not hasattr(task, '__call__'):
            raise ValueError('batch_tasks should be callable')
        # ensure variant exists
        if variant is None:
            variant = dict()

    if batch_tasks is None:
        batch_tasks = [
            dict(kwargs,
                 pre_commands=pre_commands,
                 method_call=method_call,
                 exp_name=exp_name,
                 log_dir=log_dir,
                 env=env,
                 variant=variant)
        ]

    global exp_count

    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    for task in batch_tasks:
        call = task.pop('method_call')
        data = base64.b64encode(cloudpickle.dumps(call)).decode('utf-8')
        task['args_data'] = data
        exp_count += 1

        if task.get('exp_name', None) is None:
            task['exp_name'] = '{}_{}_{:04n}'.format(exp_prefix, timestamp,
                                                     exp_count)

        if task.get('log_dir', None) is None:
            task['log_dir'] = (
                '{log_dir}/local/{exp_prefix}/{exp_name}'.format(
                    log_dir=osp.join(os.getcwd(), 'data'),
                    exp_prefix=exp_prefix.replace('_', '-'),
                    exp_name=task['exp_name']))

        if task.get('variant', None) is not None:
            variant = task.pop('variant')
            if 'exp_name' not in variant:
                variant['exp_name'] = task['exp_name']
            task['variant_data'] = base64.b64encode(
                pickle.dumps(variant)).decode('utf-8')
        elif 'variant' in task:
            del task['variant']
        task['env'] = task.get('env', dict()) or dict()
        task['env']['GARAGE_FORCE_CPU'] = str(force_cpu)

    for task in batch_tasks:
        env = task.pop('env', None)
        command = to_local_command(task,
                                   python_command=python_command,
                                   script=script)
        print(command)
        if dry:
            return
        try:
            if env is None:
                env = dict()
            subprocess.run(command,
                           shell=True,
                           env=dict(os.environ, **env),
                           check=True)
        except Exception as e:
            print(e)
            raise


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


def _shellquote(s):
    """Return a shell-escaped version of the string *s*.

    Args:
        s (str): String to shell quote.

    Returns:
        str: The shell-quoted string.

    """
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    """Return a shell-escaped version of v.

    Args:
        v (object): object to shell quote

    Returns:
        str: The shell-quoted string.

    """
    if v is None:
        return ''
    elif isinstance(v, list):
        return ' '.join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))


def to_local_command(params,
                     python_command='python',
                     script='garage.experiment.experiment_wrapper'):
    # pylint: disable=eval-used,missing-return-doc,missing-return-type-doc,missing-function-docstring
    command = python_command + ' -m ' + script

    garage_env = eval(os.environ.get('GARAGE_ENV', '{}'))
    for k, v in garage_env.items():
        command = '{}={} '.format(k, v) + command
    pre_commands = params.pop('pre_commands', None)
    post_commands = params.pop('post_commands', None)
    if pre_commands is not None or post_commands is not None:
        print('Not executing the pre_commands: ', pre_commands,
              ', nor post_commands: ', post_commands)

    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == '_name':
                    command += '  --{} {}'.format(k, _to_param_val(nv))
                else:
                    command += \
                        '  --{}_{} {}'.format(k, nk, _to_param_val(nv))
        else:
            command += '  --{} {}'.format(k, _to_param_val(v))
    return command


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

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *, function, log_dir, name, prefix, snapshot_mode,
                 snapshot_gap, archive_launch_repo):
        self.function = function
        self.log_dir = log_dir
        self.name = name
        self.prefix = prefix
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap
        self.archive_launch_repo = archive_launch_repo
        if self.function is not None:
            functools.update_wrapper(self, self.function)

    def _make_context(self, *args, **kwargs):
        """Make a context from the template information and variant args.

        Currently, all arguments should be keyword arguments.

        Args:
            args (list): Should be empty.
            kwargs (dict): Keyword arguments for the wrapped function. Will be
                logged to `variant.json`

        Returns:
            ExperimentContext: The created experiment context.

        Raises:
            ValueError: If args is not empty.

        """
        if args:
            raise ValueError('garage.experiment currently only supports '
                             'keyword arguments')
        log_dir = self.log_dir
        if log_dir is None:
            name = self.name
            if name is None:
                name = self.function.__name__
            log_dir = ('{data}/local/{prefix}/{name}'.format(
                data=os.path.join(os.getcwd(), 'data'),
                prefix=self.prefix,
                name=name))
        log_dir = _make_sequential_log_dir(log_dir)

        tabular_log_file = os.path.join(log_dir, 'progress.csv')
        text_log_file = os.path.join(log_dir, 'debug.log')
        variant_log_file = os.path.join(log_dir, 'variant.json')
        metadata_log_file = os.path.join(log_dir, 'metadata.json')

        dump_json(variant_log_file, kwargs)
        git_root_path, metadata = get_metadata()
        dump_json(metadata_log_file, metadata)
        if git_root_path and self.archive_launch_repo:
            make_launcher_archive(git_root_path=git_root_path, log_dir=log_dir)

        logger.add_output(dowel.TextOutput(text_log_file))
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(dowel.TensorBoardOutput(log_dir))
        logger.add_output(dowel.StdOutput())

        logger.push_prefix('[{}] '.format(name))
        logger.log('Logging to {}'.format(log_dir))

        return ExperimentContext(snapshot_dir=log_dir,
                                 snapshot_mode=self.snapshot_mode,
                                 snapshot_gap=self.snapshot_gap)

    def __call__(self, *args, **kwargs):
        """Wrap a function to turn it into an ExperimentTemplate.

        Note that this docstring will be overriden to match the function's
        docstring on the ExperimentTemplate once a function is passed in.

        Args:
            args (list): If no function has been set yet, must be a list
                containing a single callable.
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
            functools.update_wrapper(self, self.function)
            return self
        else:
            ctxt = self._make_context(*args, **kwargs)
            result = self.function(ctxt, *args, **kwargs)
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
                    archive_launch_repo=True):
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


    Returns:
        callable: The wrapped function.

    """
    return ExperimentTemplate(function=function,
                              log_dir=log_dir,
                              prefix=prefix,
                              name=name,
                              snapshot_mode=snapshot_mode,
                              snapshot_gap=snapshot_gap,
                              archive_launch_repo=archive_launch_repo)


def dump_json(filename, data):
    """Dump a dictionary to a file in JSON format.]

    Args:
        filename(str): Filename for the file.
        data(dict): Data to save to file.

    """
    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True, cls=LogEncoder)


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
    files_to_archive = subprocess.check_output(
        ('git', 'ls-files', '--others', '--exclude-standard', '--cached',
         '-z'),
        cwd=git_root_path).strip()
    repo_size = sum([
        os.stat(os.path.join(git_root_path, f)).st_size
        for f in files_to_archive.split(b'\0')
    ])
    if repo_size >= EIGHT_MEBIBYTES:
        warnings.warn('Archiving a launch repo larger than 8MiB. This may be '
                      'slow. Set archive_launch_repo=False in wrap_experiment '
                      'to disable this behavior.')
    archive_path = os.path.join(log_dir, 'launch_archive.tar.xz')
    subprocess.run(('tar', '--null', '--files-from', '-', '--auto-compress',
                    '--create', '--file', archive_path),
                   input=files_to_archive,
                   cwd=git_root_path,
                   check=True)


class LogEncoder(json.JSONEncoder):
    """Encoder to be used as cls in json.dump."""

    def default(self, o):
        """Perform JSON encoding.

        Args:
            o (object): Object to encode.

        Returns:
            str: Object encoded in JSON.

        """

        # Why is this method hidden? What does that mean?
        # pylint: disable=method-hidden

        if isinstance(o, type):
            return {'$class': o.__module__ + '.' + o.__name__}
        elif isinstance(o, enum.Enum):
            return {
                '$enum':
                o.__module__ + '.' + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {'$function': o.__module__ + '.' + o.__name__}
        return json.JSONEncoder.default(self, o)
