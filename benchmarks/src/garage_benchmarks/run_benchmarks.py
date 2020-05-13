#!/usr/bin/env python
"""Script for running benchmarking.

Examples:
    # List all benchmark options
    benchmark show

    # Run selected benchmarks
    benchmark run b_1 b_2 ...

"""
import inspect

import click
from garage_benchmarks import benchmark_algos
from garage_benchmarks import benchmark_auto
from garage_benchmarks import benchmark_baselines
from garage_benchmarks import benchmark_policies
from garage_benchmarks import benchmark_q_functions


@click.group()
def cli():
    """The main command group."""


@click.command()
def list():  # pylint: disable=redefined-builtin
    """List all benchmarks."""
    _echo_run_names('Algorithms', _get_runs_dict(benchmark_algos))
    _echo_run_names('Policies', _get_runs_dict(benchmark_policies))
    _echo_run_names('Baselines', _get_runs_dict(benchmark_baselines))
    _echo_run_names('Q Functions', _get_runs_dict(benchmark_q_functions))
    _echo_run_names('Automatic benchmarking', _get_runs_dict(benchmark_auto))


@click.command()
@click.argument('names', nargs=-1)
def run(names):
    """Run selected benchmarks.

    Args:
        names (tuple): Benchmark names.

    Raises:
        BadParameter: if any run name is invalid or duplicated.

    """
    if not names:
        raise click.BadParameter('Empty names!')

    if len(names) != len(set(names)):
        raise click.BadParameter('Duplicate names!')

    options = _get_all_options()

    for name in names:
        if name not in options:
            raise click.BadParameter(
                'Invalid run name! Make sure every name can be found in '
                '`garage_benchmark list`!')

    for name in names:
        options[name]()


cli.add_command(list)
cli.add_command(run)


def _get_all_options():
    """Return a dict containing all benchmark options.

    Dict of (str: obj) representing benchmark name and its function object.

    Returns:
        dict: Benchmark options.

    """
    d = {}
    d.update(_get_runs_dict(benchmark_algos))
    d.update(_get_runs_dict(benchmark_policies))
    d.update(_get_runs_dict(benchmark_baselines))
    d.update(_get_runs_dict(benchmark_q_functions))
    d.update(_get_runs_dict(benchmark_auto))
    return d


def _get_runs_dict(module):
    """Return a dict containing benchmark options of the module.

    Dict of (str: obj) representing benchmark name and its function object.

    Args:
        module (object): Module object.

    Returns:
        dict: Benchmark options of the module.

    """
    d = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name.endswith('benchmarks'):
            d[name] = obj
    return d


def _echo_run_names(header, d):
    """Echo run names to the command line.

    Args:
        header (str): The header name.
        d (dict): The dict containing benchmark options.

    """
    click.echo('-----' + header + '-----')
    for name in d:
        click.echo(name)
    click.echo()
