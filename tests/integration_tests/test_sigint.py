from enum import IntEnum
import itertools
from multiprocessing.connection import Listener
import os
import signal
import subprocess

import psutil
import pytest

scripts = [
    'tests/fixtures/algos/nop_pendulum_instrumented.py',
    'tests/fixtures/tf/trpo_pendulum_instrumented.py',
]


class ExpLifecycle(IntEnum):
    """Messages sent from InstrumentedBatchPolopt to this test."""
    START = 1
    OBTAIN_SAMPLES = 2
    PROCESS_SAMPLES = 3
    OPTIMIZE_POLICY = 4
    UPDATE_PLOT = 5
    SHUTDOWN = 6


def interrupt_experiment(experiment_script, lifecycle_stage):
    """Interrupt the experiment and verify no children processes remain."""

    args = ['python', experiment_script]
    # The pre-executed function setpgrp allows to create a process group
    # so signals are propagated to all the process in the group.
    proc = subprocess.Popen(args, preexec_fn=os.setpgrp)
    launcher_proc = psutil.Process(proc.pid)

    # This socket connects with the client in the algorithm, so we're
    # notified of the different stages in the experiment lifecycle.
    address = ('localhost', 6000)
    listener = Listener(address)
    conn = listener.accept()

    while True:
        msg = conn.recv()

        if msg == lifecycle_stage:
            # Notice that we're asking for the children of the launcher, not
            # the children of this test script, since there could be other
            # children processes attached to the process running this test
            # that are not part of the launcher.
            children = launcher_proc.children(recursive=True)
            # Remove the semaphore tracker from the list of children, since
            # we cannot stop its execution.
            for child in children:
                if any([
                        'multiprocessing.semaphore_tracker' in cmd
                        for cmd in child.cmdline()
                ]):
                    children.remove(child)
            # We append the launcher to the list of children so later we can
            # check it has died.
            children.append(launcher_proc)
            pgrp = os.getpgid(proc.pid)
            os.killpg(pgrp, signal.SIGINT)
            conn.close()
            break
    listener.close()

    # Once the signal has been sent, all children should die
    _, alive = psutil.wait_procs(children, timeout=6)

    # If any, notify the zombie and sleeping processes and fail the test
    clean_exit = True
    error_msg = ''
    for child in alive:
        error_msg += (
            str(child.as_dict(attrs=['pid', 'name', 'status', 'cmdline'])) +
            '\n')
        clean_exit = False

    error_msg = ("These processes didn't die during %s:\n" %
                 (lifecycle_stage) + error_msg)

    for child in alive:
        os.kill(child.pid, signal.SIGINT)

    assert clean_exit, error_msg


class TestSigInt:

    test_sigint_params = list(itertools.product(scripts, ExpLifecycle))

    @pytest.mark.flaky
    @pytest.mark.parametrize('experiment_script, exp_stage',
                             test_sigint_params)
    def test_sigint(self, experiment_script, exp_stage):
        """Interrupt the experiment in different stages of its lifecyle."""
        interrupt_experiment(experiment_script, exp_stage)
