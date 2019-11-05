# flake8: noqa
import base64
import collections
import datetime
import enum
import inspect
import json
import os
import pathlib
import os.path as osp
import pickle
import re
import subprocess
import uuid

import cloudpickle
import dowel
from dowel import logger
import dateutil.tz
import numpy as np

import garage.experiment

class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VariantDict(AttrDict):

    def __init__(self, d, hidden_keys):
        super(VariantDict, self).__init__(d)
        self._hidden_keys = hidden_keys

    def dump(self):
        return {k: v for k, v in self.items() if k not in self._hidden_keys}


class VariantGenerator:
    """
    Usage:

    | vg = VariantGenerator()
    | vg.add("param1", [1, 2, 3])
    | vg.add("param2", ['x', 'y'])
    | vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports noncyclic dependency among parameters:
    | vg = VariantGenerator()
    | vg.add("param1", [1, 2, 3])
    | vg.add("param2", lambda param1: [param1+1, param1+2])
    | vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()
        self._hidden_keys = []
        for k, vs, cfg in self._variants:
            if cfg.get('hide', False):
                self._hidden_keys.append(k)

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
        methods = [
            x[1].__get__(self, self.__class__) for x in methods
            if getattr(x[1], '__is_variant', False)
        ]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, '__variant_config', dict()))

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        return list(map(self.variant_dict, ret))

    def variant_dict(self, variant):
        return VariantDict(variant, self._hidden_keys)

    def to_name_suffix(self, variant):
        suffix = []
        for k, vs, cfg in self._variants:
            if not cfg.get('hide', False):
                suffix.append(k + '_' + str(variant[k]))
        return '_'.join(suffix)

    def ivariants(self):
        dependencies = list()
        for key, vals, _ in self._variants:
            if hasattr(vals, '__call__'):
                args = inspect.getfullargspec(vals).args
                if hasattr(vals, 'im_self') or hasattr(vals, '__self__'):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if not v]
            if not free_nodes:
                error_msg = 'Invalid parameter dependency: \n'
                for k, v in dependencies:
                    if v:
                        error_msg += k + ' depends on ' + ' & '.join(v) + '\n'
                raise ValueError(error_msg)
            dependencies = [(k, v) for k, v in dependencies
                            if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if not sorted_keys:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, '__call__'):
                last_val_keys = inspect.getfullargspec(last_vals).args
                if hasattr(last_vals, 'im_self') or hasattr(
                        last_vals, '__self__'):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, '__call__'):
                    last_variants = last_vals(
                        **{k: variant[k]
                           for k in last_val_keys})
                    for last_choice in last_variants:
                        yield AttrDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield AttrDict(variant, **{last_key: last_choice})


def variant(*args, **kwargs):

    def _variant(fn):
        fn.__is_variant = True
        fn.__variant_config = kwargs
        return fn

    if len(args) == 1 and isinstance(args[0], collections.Callable):
        return _variant(args[0])
    return _variant


exp_count = 0
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')


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

    """
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
            subprocess.call(command, shell=True, env=dict(os.environ, **env))
        except Exception as e:
            print(e)
            if isinstance(e, KeyboardInterrupt):
                raise


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ''
    elif isinstance(v, list):
        return ' '.join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))


def to_local_command(params,
                     python_command='python',
                     script='garage.experiment.experiment_wrapper'):
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


def choose_log_dir(exp_prefix='experiment', exp_name=None, exp_num=0,
                   start_time=None):
    """Choose a log directory for the current experiment.

    Makes use of the current directory, as well as the local time.

    """
    if start_time is None:
        start_time = datetime.datetime.now(dateutil.tz.tzlocal())
    this_timestamp = start_time.strftime('%Y_%m_%d_%H_%M_%S')
    if exp_name is None:
        exp_name = '{}_{}_{:04n}'.format(exp_prefix, this_timestamp, exp_num)

    return ('{log_dir}/local/{exp_prefix}/{exp_name}'.format(
        log_dir=osp.join(os.getcwd(), 'data'),
        exp_prefix=exp_prefix.replace('_', '-'),
        exp_name=exp_name))


def setup_experiment(log_dir=None, exp_name=None, seed=None,
                     snapshot_mode='last',
                     snapshot_gap=1,
                     tabular_log_file='progress.csv',
                     text_log_file='debug.log', params_log_file='params.json',
                     variant_log_file='variant.json',
                     params={}, variant_data={}):
    """Setup an experiment directory."""
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    rand_id = str(uuid.uuid4())[:5]
    if log_dir is None:
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        if exp_name is None:
            exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
        log_dir = choose_log_dir(exp_name=exp_name, start_time=now)
    tabular_log_file = os.path.join(log_dir, tabular_log_file)
    text_log_file = os.path.join(log_dir, text_log_file)
    params_log_file = os.path.join(log_dir, params_log_file)
    variant_log_file = os.path.join(log_dir, variant_log_file)

    dump_json(params_log_file, params)

    dump_json(variant_log_file, variant_data)

    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    logger.add_output(dowel.TensorBoardOutput(log_dir))
    logger.add_output(dowel.StdOutput())

    logger.push_prefix('[%s] ' % exp_name)

    snapshot_config = \
        garage.experiment.SnapshotConfig(snapshot_dir=log_dir,
                                         snapshot_mode=snapshot_mode,
                                         snapshot_gap=snapshot_gap)
    if seed is not None:
        garage.experiment.deterministic.set_seed(seed)
    else:
        garage.experiment.deterministic.set_seed(int(rand_id, 16))
    return snapshot_config


def dump_json(log_file, variant_data):
    """Dump the variant file."""
    pathlib.Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=LogEncoder)


class LogEncoder(json.JSONEncoder):
    """Encoder to be used as cls in json.dump."""

    def default(self, o):
        """Perform JSON encoding."""
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
