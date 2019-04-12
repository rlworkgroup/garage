# flake8: noqa
# Automated Benchmarking
[Benchmark repo](https://github.com/rlworkgroup/benchmarks)

We have an automated system that will run any benchmark file named "test_benchmark_*.py" in this folder /tests/benchmarks/
The results are displayed [here](https://rlworkgroup.github.io/benchmarks/)

# Directions
    Following these directions to output your benchmark results as a json compatible with the display.
    1. import tests.helpers as Rh
    2. Create a result_json = {}
    3. for each task: result_json [task] = Rh.create_json()
    4. Rh.write_file(result_json, 'Algo name')
