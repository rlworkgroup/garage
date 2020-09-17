# Distribute experiments across machines

This tutorial teaches you how to setup a [Prefect](https://docs.prefect.io/) +
[Dask distributed](https://distributed.dask.org/en/latest/) cluster to
run multiple garage experiments in parallel on multiple machines.

## Installation

On every machine in the cluster, install these dependencies in your virtual
environment where garage is installed. Let's assume our virtual environment is
called `my-garage-env`:

```shell script
source my-garage-env/bin/activate
# we are using distributed==2.20.0 since newer versions depend on
# cloudpickle>=1.5 which is incompatible with garage for now

pip install dask distributed==2.20.0 bokeh jupyter-server-proxy prefect
```

You can find the instructions for installing garage [here](installation.html)

## Bring up dask cluster and Prefect server

**NOTE:** this step requires Docker + docker-compose for Prefect

Select a machine to be your primary node. This machine will run dask-scheduler
and Prefect. You can also choose to run them on different machines if you need
to.

Since the scheduler runs as a foreground process, run it in a new terminal
session. If you are accessing your machine remotely over ssh, you can use tmux
or similar tools to achieve this.

### Start a dask distributed cluster

#### Start the dask scheduler

```sh
source my-garage-env/bin/activate
dask-scheduler  # listens on port 8786
```

Open a browser to [http://localhost:8787](http://localhost:8787) or
`http://<scheduler machine IP>:8787`  for the Dask Scheduler UI


#### Start dask worker

On every worker machine, in a new terminal session:

```sh
source my-garage-env/bin/activate
dask-worker <scheduler IP addr>:8786 --resources "TOKEN=1"
```

You can also run dask-worker on your primary machine.

### Start prefect

You can run the Prefect server and agent on the same machine as the
dask-scheduler for simplicity.

#### Set prefect config

Write the following config options in a file `~/.prefect/config.toml`:

```toml
backend = "server"

[server]
host = "http://<Prefect server ip>"

    [server.ui]
    # currently broken: https://github.com/PrefectHQ/prefect/issues/3254
    # for now, you can set this using the Prefect UI mentioned later
    graphql_url = "http://<Prefect server ip>/graphql"
```

#### prefect server

```sh
source my-garage-env/bin/activate
prefect server start
```

Open a browser to [http://localhost:8080](http://localhost:8080) or
`http://<Prefect machine IP>:8787` for the Prefect UI

#### prefect agent

```sh
source my-garage-env/bin/activate
prefect agent start
```

### Defining your workflow

The simplest way to wrap your experiment in a Prefect task is to use the `@task`
decorator. You can see more details and alternative ways on the [Prefect docs
page for Tasks](https://docs.prefect.io/core/concepts/tasks.html)

Then you can use the [`Flow` API from Prefect](https://docs.prefect.io/core/concepts/flows.html#overview)
to add tasks to your workflow.

```py
from garage import wrap_experiment

from prefect import task, Flow
from prefect.engine.executors import DaskExecutor
from prefect.environments import LocalEnvironment
from prefect.utilities.logging import get_logger

logger = get_logger('dask-example')

# replace localhost with IP of your dask-scheduler machine
executor = DaskExecutor(address='tcp://localhost:8786')
environment = LocalEnvironment(executor=executor)

@task(tags=['dask-resource:TOKEN=1'])
@wrap_experiment
def my_experiment_a(ctxt=None, seed=1):
    ...

@task(tags=['dask-resource:TOKEN=1'])
@wrap_experiment
def my_experiment_b(ctxt=None, seed=1):
    ...

# dask-example is the name given to the flow
with Flow('dask-example', environment=environment) as flow:
    my_experiment_a()
    my_experiment_b()
```

### Register your job with prefect

```sh
source my-garage-env/bin/activate
# Create a new project, let's call it Dask
prefect create project "Dask"
prefect register flow --file hello_prefect.py --project "Dask"
```

Your workflow should now show up in the web UI you launched earlier.
Alternatively, you can open a browser to the
"Flow: http://localhost/8080/flow/..." outputted by prefect.

Repeated registrations create a new sequential version in the DB.

### Run your job with prefect
From the Web UI, click "Quick Run" on your Flow record. This will schedule your
tasks onto the Dask Distributed cluster.

### Check out your job

#### Console logs
You should see some console logs from your tasks in the terminal windows for
your dask workers.

#### dask.distributed dashboard
Open the Dask scheduler UI at `http://<scheduler machine IP>:8787`

You will see a Gantt chart of the tasks being executed, and some bar charts
representing queued tasks. Clicking on a task's block in the Gantt chart will
show you a sampled profile of its execution.

On the "Info" tab you can click through to dashboards for individual workers.
These will show you resource accounting and task status on the worker, among
other stats.

#### Prefect UI
Look at [http://localhost:8080](http://localhost:8080).

The "Logs" tab of prefect will show you streaming log output from each of your
tasks. It will also show you a graph representation of your workflow and a
Gantt chart of the tasks, similar to dask.

Each registration creates a new version of the same (work)flow. Each run of that
`(flow, version)` creates a unique run record, i.e. runs are unique keys of the
form `(flow, version, run_id)`.


## More info on `dask-worker`s and resources

### Resources
Resources are completely arbitrary and abstract. That is, you could replace
TOKEN with GPU or CPU or CHICKEN. A resource is just a labeled quantity which
tasks can declare they require exclusively. A task's resource claims decrement
available resources on the worker process, and the worker process will not
schedule any task which would allow its resource counters to go negative.

**IMPORTANT:** Resource declarations apply PER PROCESS regardless of invocation
That means the following two invocations are equivalent for resource purposes.

One worker, two processes:

```sh
dask-worker localhost:8786 --nprocs 2 --resources TOKEN=1
```

Two workers, one process each:

```
dask-worker localhost:8786 --resources TOKEN=1 &  # default --nprocs 1
dask-worker localhost:8786 --resources TOKEN=1 &
```

### Dask worker processes, threads, and tasks
`dask-worker` by default launches with 1 process and as many threads as your
processor has hyper-threads. A task consumes a single thread. The number of
processes and threads-per-process can be controlled by the `--nprocs` and
`--nthreads` flags respectively.

Garage is a multi-threaded multi-process application, so we would want to
ensure that all experiments run in their own process, and that each of those
processes has 2 worker threads (1 for garage and 1 for servicing the dask API).
To avoid overloading CPUs, GPUs, and memory, we would have to account for that
on a per-experiment basis.

Launch worker:

```sh
NCPU="$(nproc)"
NMEM="$(awk '/MemFree/ { printf $2 }' /proc/meminfo)"

dask-worker localhost:8786 \
  --nprocs "${NCPU}" \
  --nthreads 2 \
  --resources "PROCESS=${NCPU},CPU=${NCPU},MEMORY=${NMEM}"
```

Resource tags:

```python
@prefect.task(tags=['dask-resource:PROCESS=1'])
@garage.wrap_experiment
def my_experiment(ctxt, t):
    ...
```

### Strategies for GPUs
In the case of GPUs, assuming we want GPU exclusivity and locality, we could
start 1 worker per GPU instead and equally-divide the processes. We can use
`CUDA_VISIBLE_DEVICES` to enforce exclusivity.

If you don't want exclusivity, dask will happily schedule fractional GPUs.

Experiment-exclusive GPUs, with fixed CPU and memory allocations per-worker.

Launch worker:

```sh
NCPU="$(nproc)"
NGPU=4
NCPU_PER_GPU="$(( $NCPU / $NGPU ))"
NMEM_PER_GPU="$(( $NMEM / $NGPU ))"

for i in {0..3}; do
  CUDA_VISIBLE_DEVICES=i dask-worker localhost:8786 \
    --nprocs "${NCPU_PER_GPU}" \
    --nthreads 2 \
    --resources "PROCESS=${NCPU},GPU=1,CPU=${NCPU_PER_GPU},MEMORY=${NMEM_PER_GPU}" &
done
```

Resource tags:

```python
# 1 GPU, ~10GB RAM, 18 CPU threads
@prefect.task(tags=[
    'dask-resource:PROCESS=1',
    'dask-resource:MEMORY=10e9',
    'dask-resource:GPU=1',
    'dask-resource:CPU=18',
])
@garage.wrap_experiment
def my_experiment(ctxt, t):
    ...
```
