import numpy as np


class MultiTaskBaseline:

    def __init__(self, env_spec, n_tasks, baseline_cls, reg_coeff=1e-5):
        self._cur_task = None
        self._baselines = [
            baseline_cls(env_spec=env_spec, reg_coeff=reg_coeff)
                for _ in range(n_tasks)
        ]

    def get_param_values(self):
        return self._baselines[self._cur_task]._coeffs

    def set_param_values(self, val):
        self._baselines[self._cur_task]._coeffs = val

    def fit(self, task_paths):
        assert len(task_paths) == len(self._baselines)

        for paths, baseline in zip(task_paths, self._baselines):
            baseline.fit(paths)

    def predict(self, path):
        return self._baselines[self._cur_task].predict(path)

    def set_task(self, task):
        self._cur_task = task

