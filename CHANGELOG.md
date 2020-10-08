# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2020.06.3
- Fixed
  * PyTorch 1.7 support ([#1934](https://github.com/rlworkgroup/garage/pull/1934))
  * `LocalRunner` ignores `worker_cls` attribute of algorithms ([#1984](https://github.com/rlworkgroup/garage/pull/1984))
  * `mujoco_py` versions greater than v2.0.2.8 are incompatible with some GCC versions in conda ([#2000](https://github.com/rlworkgroup/garage/pull/2000))
  * MTSAC not learning because it corrupts the termination signal by wrapping with `GarageEnv` twice ([#2029](https://github.com/rlworkgroup/garage/pull/2029))
  * MTSAC does not respect `max_episode_length_eval` hyperparameter ([#2029](https://github.com/rlworkgroup/garage/pull/2029))
  * MTSAC MetaWorld examples do not use the correct number of tasks ([#2029](https://github.com/rlworkgroup/garage/pull/2029))
  * MTSAC now supports a separate `max_episode_length` for evalaution via the `max_episode_length_eval` hyperparameter ([#2029](https://github.com/rlworkgroup/garage/pull/2029))
  * MTSAC MetaWorld MT50 example used an incorrect `max_episode_length` ([#2029](https://github.com/rlworkgroup/garage/pull/2029))

## 2020.06.2
- Fixed
  * Better parameters for example `her_ddpg_fetchreach` ([#1763](https://github.com/rlworkgroup/garage/pull/1763))
  * Ensure determinism in TensorFlow by using `tfp.SeedStream` ([#1821](https://github.com/rlworkgroup/garage/pull/1821))
  * Broken rendering of MuJoCo environments to pixels in the NVIDIA Docker container ([#1838](https://github.com/rlworkgroup/garage/pull/1838))
  * Enable cudnn in the NVIDIA Docker container ([#1840](https://github.com/rlworkgroup/garage/pull/1840))
  * Bug in `DiscreteQfDerivedPolicy` in which parameters were not returned ([#1847](https://github.com/rlworkgroup/garage/pull/1847))
  * Populate `TimeLimit.truncated` at every step when using `gym.Env` ([#1852](https://github.com/rlworkgroup/garage/pull/1852))
  * Bug in which parameters where not copied when TensorFlow primitives are `clone()`ed ([#1855](https://github.com/rlworkgroup/garage/pull/1855))
  * Typo in the `Makefile` target `run-nvidia` ([#1914](https://github.com/rlworkgroup/garage/pull/1914))

## 2020.06.1
- Fixed
  * Pipenv fails to resolve a stable dependency set because of excessively-narrow dependencies in tensorflow-probability ([#1721](https://github.com/rlworkgroup/garage/pull/1721))
  * Bug which prevented `rollout` from running policies deterministically ([#1714](https://github.com/rlworkgroup/garage/pull/1714))

## 2020.06.0

### Added
- Algorithms
  * PPO in PyTorch (
    [#905](https://github.com/rlworkgroup/garage/pull/905),
    [#1188](https://github.com/rlworkgroup/garage/pull/1188))
  * TRPO in PyTorch (
    [#1018](https://github.com/rlworkgroup/garage/pull/1018),
    [#1053](https://github.com/rlworkgroup/garage/pull/1053),
    [#1186](https://github.com/rlworkgroup/garage/pull/1186))
  * MAML in PyTorch (
    [#1128](https://github.com/rlworkgroup/garage/pull/1128),
    [#1136](https://github.com/rlworkgroup/garage/pull/1136),
    [#1214](https://github.com/rlworkgroup/garage/pull/1214),
    [#1234](https://github.com/rlworkgroup/garage/pull/1234),
    [#1283](https://github.com/rlworkgroup/garage/pull/1283))
  * RL2 in TensorFlow (
    [#1127](https://github.com/rlworkgroup/garage/pull/1127),
    [#1175](https://github.com/rlworkgroup/garage/pull/1175),
    [#1189](https://github.com/rlworkgroup/garage/pull/1189),
    [#1190](https://github.com/rlworkgroup/garage/pull/1190),
    [#1195](https://github.com/rlworkgroup/garage/pull/1195),
    [#1231](https://github.com/rlworkgroup/garage/pull/1231))
  * PEARL in PyTorch (
    [#1059](https://github.com/rlworkgroup/garage/pull/1059),
    [#1124](https://github.com/rlworkgroup/garage/pull/1124),
    [#1218](https://github.com/rlworkgroup/garage/pull/1218),
    [#1316](https://github.com/rlworkgroup/garage/pull/1316),
    [#1374](https://github.com/rlworkgroup/garage/pull/1374))
  * SAC in PyTorch ([#1235](https://github.com/rlworkgroup/garage/pull/1235))
  * MTSAC in PyTorch ([#1332](https://github.com/rlworkgroup/garage/pull/1332))
  * Task Embeddings in TensorFlow (
    [#1168](https://github.com/rlworkgroup/garage/pull/1168),
    [#1169](https://github.com/rlworkgroup/garage/pull/1169),
    [#1167](https://github.com/rlworkgroup/garage/pull/1167))
- Samplers
  * New Sampler API, with efficient multi-env and multi-policy support (
    [#881](https://github.com/rlworkgroup/garage/pull/881),
    [#1153](https://github.com/rlworkgroup/garage/pull/1153),
    [#1319](https://github.com/rlworkgroup/garage/pull/1319))
  * `garage.sampler.LocalSampler`, which uses the main process to sample (
    [#1133](https://github.com/rlworkgroup/garage/pull/1133),
    [#1156](https://github.com/rlworkgroup/garage/pull/1156))
  * Reworked `garage.sampler.RaySampler` to use new API (
    [#1133](https://github.com/rlworkgroup/garage/pull/1133),
    [#1134](https://github.com/rlworkgroup/garage/pull/1134))
  * `garage.sampler.MultiprocessingSampler` ([#1298](https://github.com/rlworkgroup/garage/pull/1298))
  * `garage.sampler.VecWorker`, a replacement for VecEnvExecutor ([#1311](https://github.com/rlworkgroup/garage/pull/1311))
- APIs
  * `garage.TrajectoryBatch` data type (
    [#1058](https://github.com/rlworkgroup/garage/pull/1058),
    [#1065](https://github.com/rlworkgroup/garage/pull/1065),
    [#1132](https://github.com/rlworkgroup/garage/pull/1132),
    [#1154](https://github.com/rlworkgroup/garage/pull/1154))
  * `garage.TimeStep` data type (
    [#1114](https://github.com/rlworkgroup/garage/pull/1114),
    [#1221](https://github.com/rlworkgroup/garage/pull/1221))
  * `garage.TimeStepBatch` data type ([#1529](https://github.com/rlworkgroup/garage/pull/1529))
  * `garage.log_performance` (
    [#1116](https://github.com/rlworkgroup/garage/pull/1116),
    [#1142](https://github.com/rlworkgroup/garage/pull/1142),
    [#1159](https://github.com/rlworkgroup/garage/pull/1159))
  * `garage.np.algos.MetaRLAlgorithm` ([#1142](https://github.com/rlworkgroup/garage/pull/1142))
  * `garage.experiment.MetaEvaluator` (
    [#1142](https://github.com/rlworkgroup/garage/pull/1142),
    [#1152](https://github.com/rlworkgroup/garage/pull/1152),
    [#1227](https://github.com/rlworkgroup/garage/pull/1227))
  * `garage.log_multitask_performance` ([#1192](https://github.com/rlworkgroup/garage/pull/1192))
  * `garage.torch.distributions.TanhNormal` ([#1140](https://github.com/rlworkgroup/garage/pull/1140))
  * `garage.torch.policies.TanhGaussianMLPPolicy` ([#1176](https://github.com/rlworkgroup/garage/pull/1176))
  * `garage.experiment.wrap_experiment` to replace `run_experiment` with several new features (
    [#1100](https://github.com/rlworkgroup/garage/pull/1100),
    [#1155](https://github.com/rlworkgroup/garage/pull/1155),
    [#1160](https://github.com/rlworkgroup/garage/pull/1160),
    [#1164](https://github.com/rlworkgroup/garage/pull/1164),
    [#1249](https://github.com/rlworkgroup/garage/pull/1249),
    [#1258](https://github.com/rlworkgroup/garage/pull/1258),
    [#1281](https://github.com/rlworkgroup/garage/pull/1281),
    [#1396](https://github.com/rlworkgroup/garage/pull/1396),
    [#1482](https://github.com/rlworkgroup/garage/pull/1482))
  * `garage.torch.q_functions.ContinuousCNNQFunction` ([#1326](https://github.com/rlworkgroup/garage/pull/1326))
  * PyTorch support for non-linearities with parameters ([#928](https://github.com/rlworkgroup/garage/pull/928),
  * `garage.torch.value_function.GaussianMLPValueFunction` (
    [#1317](https://github.com/rlworkgroup/garage/pull/1317))
  * Simpler PyTorch policy API ([#1528](https://github.com/rlworkgroup/garage/pull/1528))
  * `garage.envs.TaskOnehotWrapper` ([#1157](https://github.com/rlworkgroup/garage/pull/1157))
- HalfCheetah meta environments (
    [#1108](https://github.com/rlworkgroup/garage/pull/1108),
    [#1131](https://github.com/rlworkgroup/garage/pull/1131),
    [#1216](https://github.com/rlworkgroup/garage/pull/1216),
    [#1385](https://github.com/rlworkgroup/garage/pull/1385))
- PyTorch GPU support ([#1182](https://github.com/rlworkgroup/garage/pull/1182))
- PyTorch deterministic support ([#1063](https://github.com/rlworkgroup/garage/pull/1063))
- Support for running Meta-RL algorithms on MetaWorld benchmarks (
  [#1306](https://github.com/rlworkgroup/garage/pull/1306))
- Examples for running MetaWorld benchmarks (
  [#1010](https://github.com/rlworkgroup/garage/pull/1010),
  [#1263](https://github.com/rlworkgroup/garage/pull/1263),
  [#1265](https://github.com/rlworkgroup/garage/pull/1265),
  [#1265](https://github.com/rlworkgroup/garage/pull/1265),
  [#1241](https://github.com/rlworkgroup/garage/pull/1241),
  [#1232](https://github.com/rlworkgroup/garage/pull/1232),
  [#1327](https://github.com/rlworkgroup/garage/pull/1327),
  [#1351](https://github.com/rlworkgroup/garage/pull/1351),
  [#1393](https://github.com/rlworkgroup/garage/pull/1393))
- Improved off-policy evaluation (
  [#1139](https://github.com/rlworkgroup/garage/pull/1139),
  [#1279](https://github.com/rlworkgroup/garage/pull/1279),
  [#1331](https://github.com/rlworkgroup/garage/pull/1331),
  [#1384](https://github.com/rlworkgroup/garage/pull/1384))

### Changed
- Allow TensorFlow 2 (or TF >=1.14) (
  [#1309](https://github.com/rlworkgroup/garage/pull/1309),
  [#1563](https://github.com/rlworkgroup/garage/pull/1563))
- Require Torch 1.4.0 (
  [#1335](https://github.com/rlworkgroup/garage/pull/1335),
  [#1361](https://github.com/rlworkgroup/garage/pull/1361))
- Ensure TF and torch are optional ([#1510](https://github.com/rlworkgroup/garage/pull/1510))
- Update gym to 0.15.4 (
  [#1098](https://github.com/rlworkgroup/garage/pull/1098),
  [#1158](https://github.com/rlworkgroup/garage/pull/1158))
- Rename `baseline` to `value_function` ([#1275](https://github.com/rlworkgroup/garage/pull/1275))
- Make `runner._sampler` optional ([#1394](https://github.com/rlworkgroup/garage/pull/1394))
- Make ExplorationStrategies a type of Policy ([#1397](https://github.com/rlworkgroup/garage/pull/1397))
- Use `garage.replay_buffer.PathBuffer` in off-policy algos (
  [#1173](https://github.com/rlworkgroup/garage/pull/1173),
  [#1433](https://github.com/rlworkgroup/garage/pull/1433))
- Deprecated `run_experiment` (
    [#1370](https://github.com/rlworkgroup/garage/pull/1370),
    [#1412](https://github.com/rlworkgroup/garage/pull/1412))
- Deprecated old-style samplers ([#1369](https://github.com/rlworkgroup/garage/pull/1369))
- Refactor TensorFlow to use tfp.distribution (
  [#1073](https://github.com/rlworkgroup/garage/pull/1073),
  [#1356](https://github.com/rlworkgroup/garage/pull/1356),
  [#1357](https://github.com/rlworkgroup/garage/pull/1357),
  [#1410](https://github.com/rlworkgroup/garage/pull/1410),
  [#1456](https://github.com/rlworkgroup/garage/pull/1456),
  [#1444](https://github.com/rlworkgroup/garage/pull/1444),
  [#1554](https://github.com/rlworkgroup/garage/pull/1554),
  [#1569](https://github.com/rlworkgroup/garage/pull/1569))
- Set TotalEnvSteps as the default Tensorboard x-axis (
  [#1017](https://github.com/rlworkgroup/garage/pull/1017),
  [#1069](https://github.com/rlworkgroup/garage/pull/1069))
- Update dependencies for docs ([#1383](https://github.com/rlworkgroup/garage/pull/1383))
- New optimizer_args TensorFlow interface ([#1496](https://github.com/rlworkgroup/garage/pull/1496))
- Move LocalTFRunner to garage.experiment ([#1513](https://github.com/rlworkgroup/garage/pull/1513))
- Implement HER using PathBuffer (
  [#1282](https://github.com/rlworkgroup/garage/pull/1282)
  [#1505](https://github.com/rlworkgroup/garage/pull/1505))
- Change CNN API to use tuples for defining kernels ([#1515](https://github.com/rlworkgroup/garage/pull/1515))
- Many documentation improvements (
  [#1056](https://github.com/rlworkgroup/garage/pull/1056),
  [#1065](https://github.com/rlworkgroup/garage/pull/1065),
  [#1120](https://github.com/rlworkgroup/garage/pull/1120),
  [#1266](https://github.com/rlworkgroup/garage/pull/1266),
  [#1327](https://github.com/rlworkgroup/garage/pull/1327),
  [#1413](https://github.com/rlworkgroup/garage/pull/1413),
  [#1429](https://github.com/rlworkgroup/garage/pull/1429),
  [#1451](https://github.com/rlworkgroup/garage/pull/1451),
  [#1481](https://github.com/rlworkgroup/garage/pull/1481),
  [#1484](https://github.com/rlworkgroup/garage/pull/1484))
- Eliminate use of "base" module name ([#1403](https://github.com/rlworkgroup/garage/pull/1403))
- Significant improvements to benchmarking (
  [#1271](https://github.com/rlworkgroup/garage/pull/1271)
  [#1291](https://github.com/rlworkgroup/garage/pull/1291),
  [#1306](https://github.com/rlworkgroup/garage/pull/1306),
  [#1307](https://github.com/rlworkgroup/garage/pull/1307),
  [#1310](https://github.com/rlworkgroup/garage/pull/1310),
  [#1320](https://github.com/rlworkgroup/garage/pull/1320),
  [#1368](https://github.com/rlworkgroup/garage/pull/1368),
  [#1380](https://github.com/rlworkgroup/garage/pull/1380),
  [#1409](https://github.com/rlworkgroup/garage/pull/1409))
- Refactor benchmarks into a separate module (
  [#1395](https://github.com/rlworkgroup/garage/pull/1395),
  [#1402](https://github.com/rlworkgroup/garage/pull/1402),
  [#1400](https://github.com/rlworkgroup/garage/pull/1400),
  [#1411](https://github.com/rlworkgroup/garage/pull/1411),
  [#1408](https://github.com/rlworkgroup/garage/pull/1408),
  [#1416](https://github.com/rlworkgroup/garage/pull/1416),
  [#1414](https://github.com/rlworkgroup/garage/pull/1414),
  [#1432](https://github.com/rlworkgroup/garage/pull/1432))

### Removed
- Dependencies:
  * matplotlib (moved to dev) ([#1083](https://github.com/rlworkgroup/garage/pull/1083))
  * atari-py ([#1194](https://github.com/rlworkgroup/garage/pull/1194))
  * gtimer, pandas, rlkit, seaborn (moved to benchmarks) ([#1325](https://github.com/rlworkgroup/garage/pull/1325))
  * pyprind ([#1495](https://github.com/rlworkgroup/garage/pull/1495))
- `RLAlgorithm.get_itr_snapshot` ([#1054](https://github.com/rlworkgroup/garage/pull/1054))
- `garage.misc.nb_utils` ([#1288](https://github.com/rlworkgroup/garage/pull/1288))
- `garage.np.regressors` ([#1493](https://github.com/rlworkgroup/garage/pull/1493))
- `garage.np.BatchPolopt` (
  [#1486](https://github.com/rlworkgroup/garage/pull/1486),
  [#1492](https://github.com/rlworkgroup/garage/pull/1492))
- `garage.misc.prog_bar_counter` ([#1495](https://github.com/rlworkgroup/garage/pull/1495))
- `garage.tf.envs.TfEnv` ([#1443](https://github.com/rlworkgroup/garage/pull/1443))
- `garage.tf.BatchPolopt` ([#1504](https://github.com/rlworkgroup/garage/pull/1504))
- `garage.np.OffPolicyRLAlgorithm` ([#1552](https://github.com/rlworkgroup/garage/pull/1552))

### Fixed
- Bug where `GymEnv` did not pickle ([#1029](https://github.com/rlworkgroup/garage/pull/1029))
- Bug where `VecEnvExecutor` conflated terminal state and time limit signal (
  [#1178](https://github.com/rlworkgroup/garage/pull/1178),
  [#1570](https://github.com/rlworkgroup/garage/pull/1570))
- Bug where plotter window was opened multiple times ([#1253](https://github.com/rlworkgroup/garage/pull/1253))
- Bug where TF plotter used main policy on separate thread ([#1270](https://github.com/rlworkgroup/garage/pull/1270))
- Workaround gym timelimit and terminal state conflation ([#1118](https://github.com/rlworkgroup/garage/pull/1118))
- Bug where pixels weren't normalized correctly when using CNNs (
  [#1236](https://github.com/rlworkgroup/garage/pull/1236),
  [#1419](https://github.com/rlworkgroup/garage/pull/1419))
- Bug where `garage.envs.PointEnv` did not step correctly ([#1165](https://github.com/rlworkgroup/garage/pull/1165))
- Bug where sampler workers crashed in non-Deterministic mode ([#1567](https://github.com/rlworkgroup/garage/pull/1567))
- Use cloudpickle in old-style samplers to handle lambdas ([#1371](https://github.com/rlworkgroup/garage/pull/1371))
- Bug where workers where not shut down after running a resumed algorithm ([#1293](https://github.com/rlworkgroup/garage/pull/1293))
- Non-PyPI dependencies, which blocked using pipenv and poetry ([#1247](https://github.com/rlworkgroup/garage/pull/1247))
- Bug where TensorFlow paramter setting didn't work across differently named policies ([#1355](https://github.com/rlworkgroup/garage/pull/1355))
- Bug where advantages where computed incorrectly in PyTorch ([#1197](https://github.com/rlworkgroup/garage/pull/1197))
- Bug where TF plotter was used in LocalRunner ([#1267](https://github.com/rlworkgroup/garage/pull/1267))
- Worker processes are no longer started unnecessarily ([#1006](https://github.com/rlworkgroup/garage/pull/1006))
- All examples where fixed and are now tested ([#1009](https://github.com/rlworkgroup/garage/pull/1009))

## 2019.10.3

### Fixed
- Better parameters for example `her_ddpg_fetchreach` ([#1764](https://github.com/rlworkgroup/garage/pull/1764))
- Bug in `DiscreteQfDerivedPolicy` in which parameters were not returned ([#1847](https://github.com/rlworkgroup/garage/pull/1847))
- Bug which made it impossible to evaluate stochastic policies deterministically ([#1715](https://github.com/rlworkgroup/garage/pull/1715))

## 2019.10.2

### Fixed
- Use a GitHub Token in the CI to retrieve packages to avoid hitting GitHub API rate limit ([#1250](https://github.com/rlworkgroup/garage/pull/1250))
- Avoid installing dev extra dependencies during the conda check ([#1296](https://github.com/rlworkgroup/garage/pull/1296))
- Install `dm_control` from PyPI ([#1406](https://github.com/rlworkgroup/garage/pull/1406))
- Pin tfp to 0.8.x to avoid breaking pipenv ([#1480](https://github.com/rlworkgroup/garage/pull/1480))
- Force python 3.5 in CI ([#1522](https://github.com/rlworkgroup/garage/pull/1522))
- Separate terminal and completion signal in vectorized sampler ([#1581](https://github.com/rlworkgroup/garage/pull/1581))
- Disable certicate check for roboti.us ([#1595](https://github.com/rlworkgroup/garage/pull/1595))
- Fix `advantages` shape in `compute_advantage()` in torch tree ([#1209](https://github.com/rlworkgroup/garage/pull/1209))
- Fix plotting using tf.plotter ([#1292](https://github.com/rlworkgroup/garage/pull/1292))
- Fix duplicate window rendering when using garage.Plotter ([#1299](https://github.com/rlworkgroup/garage/pull/1299))
- Fix setting garage.model parameters ([#1363](https://github.com/rlworkgroup/garage/pull/1363))
- Fix two example jupyter notebook ([#1584](https://github.com/rlworkgroup/garage/pull/1584))
- Fix collecting samples in `RaySampler` ([#1583](https://github.com/rlworkgroup/garage/pull/1583))

## 2019.10.1

### Added
- Integration tests which cover all example scripts (
  [#1078](https://github.com/rlworkgroup/garage/pull/1078),
  [#1090](https://github.com/rlworkgroup/garage/pull/1090))
- Deterministic mode support for PyTorch ([#1068](https://github.com/rlworkgroup/garage/pull/1068))
- Install script support for macOS 10.15.1 ([#1051](https://github.com/rlworkgroup/garage/pull/1051))
- PyTorch modules now support either functions or modules for specifying their non-linearities ([#1038](https://github.com/rlworkgroup/garage/pull/1038))

### Fixed
- Errors in the documentation on implementing new algorithms ([#1074](https://github.com/rlworkgroup/garage/pull/1074))
- Broken example for DDPG+HER in TensorFlow ([#1070](https://github.com/rlworkgroup/garage/pull/1070))
- Error in the documentation for using garage with conda ([#1066](https://github.com/rlworkgroup/garage/pull/1066))
- Broken pickling of environment wrappers ([#1061](https://github.com/rlworkgroup/garage/pull/1061))
- `garage.torch` was not included in the PyPI distribution ([#1037](https://github.com/rlworkgroup/garage/pull/1037))
- A few broken examples for `garage.tf` ([#1032](https://github.com/rlworkgroup/garage/pull/1032))


## 2019.10.0

### Added
- Algorithms
  * (D)DQN in TensorFlow ([#582](https://github.com/rlworkgroup/garage/pull/582))
  * Maximum-entropy and entropy regularization for policy gradient algorithms in
    TensorFlow ([#632](https://github.com/rlworkgroup/garage/pull/632))
  * DDPG in PyTorch ([#815](https://github.com/rlworkgroup/garage/pull/815))
  * VPG (i.e. policy gradients) in PyTorch ([#883](https://github.com/rlworkgroup/garage/pull/883))
  * TD3 in TensorFlow ([#458](https://github.com/rlworkgroup/garage/pull/458))
- APIs
  * Runner API for executing experiments and `LocalRunner` implementation for
    executing them on the local machine (
    [#541](https://github.com/rlworkgroup/garage/pull/541),
    [#593](https://github.com/rlworkgroup/garage/pull/593),
    [#602](https://github.com/rlworkgroup/garage/pull/602),
    [#816](https://github.com/rlworkgroup/garage/pull/816),
    )
  * New Logger API, provided by a sister project [dowel](https://github.com/rlworkgroup/dowel) ([#464](https://github.com/rlworkgroup/garage/pull/464), [#660](https://github.com/rlworkgroup/garage/pull/660))
- Environment wrappers for pixel-based algorithms, especially DQN ([#556](https://github.com/rlworkgroup/garage/pull/556))
- Example for how to use garage with Google Colab ([#476](https://github.com/rlworkgroup/garage/pull/476))
- Advantage normalization for recurrent policies in TF ([#626](https://github.com/rlworkgroup/garage/pull/626))
- PyTorch support ([#725](https://github.com/rlworkgroup/garage/pull/725), [#764](https://github.com/rlworkgroup/garage/pull/764))
- Autogenerated API docs on [garage.readthedocs.io](https://garage.readthedocs.io/en/latest/py-modindex.html) ([#802](https://github.com/rlworkgroup/garage/pull/802))
- GPU version of the pip package ([#834](https://github.com/rlworkgroup/garage/pull/834))
- PathBuffer, a trajectory-oriented replay buffer ([#838](https://github.com/rlworkgroup/garage/pull/838))
- RaySampler, a remote and/or multiprocess sampler based on ray ([#793](https://github.com/rlworkgroup/garage/pull/793))
- Garage is now distributed on PyPI ([#870](https://github.com/rlworkgroup/garage/pull/870))
- `rollout` option to only sample policies deterministically ([#896](https://github.com/rlworkgroup/garage/pull/896))
- MultiEnvWrapper, which wraps multiple `gym.Env` environments into a discrete
  multi-task environment ([#946](https://github.com/rlworkgroup/garage/pull/946))

### Changed
- Optimized Dockerfiles for fast rebuilds ([#557](https://github.com/rlworkgroup/garage/pull/557))
- Random seed APIs moved to `garage.experiment.deterministic` ([#578](https://github.com/rlworkgroup/garage/pull/578))
- Experiment wrapper script is now an ordinary module ([#586](https://github.com/rlworkgroup/garage/pull/586))
- numpy-based modules and algorithms moved to `garage.np` ([#604](https://github.com/rlworkgroup/garage/pull/604))
- Algorithm constructors now use `EnvSpec` rather than `gym.Env` ([#575](https://github.com/rlworkgroup/garage/pull/575))
- Snapshotter API moved from `garage.logger` to `garage.experiment` ([#658](https://github.com/rlworkgroup/garage/pull/658))
- Moved `process_samples` API from the Sampler to algorithms ([#652](https://github.com/rlworkgroup/garage/pull/652))
- Updated Snapshotter API ([#699](https://github.com/rlworkgroup/garage/pull/699))
- Updated Resume API ([#777](https://github.com/rlworkgroup/garage/pull/777))
- All algorithms now have a default sampler ([#832](https://github.com/rlworkgroup/garage/pull/832))
- Experiment lauchers now require an explicit `snapshot_config` to their
  `run_task` function ([#860](https://github.com/rlworkgroup/garage/pull/860))
- Various samplers moved from `garage.tf.sampler` to `garage.sampler` ([#836](https://github.com/rlworkgroup/garage/pull/836),
  [#840](https://github.com/rlworkgroup/garage/pull/840))
- Dockerfiles are now based on Ubuntu 18.04 LTS by default ([#763](https://github.com/rlworkgroup/garage/pull/763))
- `dm_control` is now an optional dependency, installed using the extra
  `garage[dm_control]` ([#828](https://github.com/rlworkgroup/garage/pull/828))
- MuJoCo is now an optional dependency, installed using the extra
  `garage[mujoco]` ([#848](https://github.com/rlworkgroup/garage/pull/828))
- Samplers no longer flatten observations and actions ([#930](https://github.com/rlworkgroup/garage/pull/930),
  [#938](https://github.com/rlworkgroup/garage/pull/938),
  [#967](https://github.com/rlworkgroup/garage/pull/967))
- Implementations, tests, and benchmarks for all TensorFlow primitives, which
  are now based on `garage.tf.Model` ([#574](https://github.com/rlworkgroup/garage/pull/574),
  [#606](https://github.com/rlworkgroup/garage/pull/606),
  [#615](https://github.com/rlworkgroup/garage/pull/615),
  [#616](https://github.com/rlworkgroup/garage/pull/616),
  [#618](https://github.com/rlworkgroup/garage/pull/618),
  [#641](https://github.com/rlworkgroup/garage/pull/641),
  [#642](https://github.com/rlworkgroup/garage/pull/642),
  [#656](https://github.com/rlworkgroup/garage/pull/656),
  [#662](https://github.com/rlworkgroup/garage/pull/662),
  [#668](https://github.com/rlworkgroup/garage/pull/668),
  [#672](https://github.com/rlworkgroup/garage/pull/672),
  [#677](https://github.com/rlworkgroup/garage/pull/677),
  [#730](https://github.com/rlworkgroup/garage/pull/730),
  [#722](https://github.com/rlworkgroup/garage/pull/722),
  [#765](https://github.com/rlworkgroup/garage/pull/765),
  [#855](https://github.com/rlworkgroup/garage/pull/855),
  [#878](https://github.com/rlworkgroup/garage/pull/878),
  [#888](https://github.com/rlworkgroup/garage/pull/888),
  [#898](https://github.com/rlworkgroup/garage/pull/898),
  [#892](https://github.com/rlworkgroup/garage/pull/892),
  [#897](https://github.com/rlworkgroup/garage/pull/897),
  [#893](https://github.com/rlworkgroup/garage/pull/893),
  [#890](https://github.com/rlworkgroup/garage/pull/890),
  [#903](https://github.com/rlworkgroup/garage/pull/903),
  [#916](https://github.com/rlworkgroup/garage/pull/916),
  [#891](https://github.com/rlworkgroup/garage/pull/891),
  [#922](https://github.com/rlworkgroup/garage/pull/922),
  [#931](https://github.com/rlworkgroup/garage/pull/931),
  [#933](https://github.com/rlworkgroup/garage/pull/933),
  [#906](https://github.com/rlworkgroup/garage/pull/906),
  [#945](https://github.com/rlworkgroup/garage/pull/945),
  [#944](https://github.com/rlworkgroup/garage/pull/944),
  [#943](https://github.com/rlworkgroup/garage/pull/943),
  [#972](https://github.com/rlworkgroup/garage/pull/972))
- Dependency upgrades:
  * mujoco-py to 2.0 ([#661](https://github.com/rlworkgroup/garage/pull/661))
  * gym to 0.12.4 ([#661](https://github.com/rlworkgroup/garage/pull/661))
  * dm_control to 7a36377879c57777e5d5b4da5aae2cd2a29b607a ([#661](https://github.com/rlworkgroup/garage/pull/661))
  * akro to 0.0.6 ([#796](https://github.com/rlworkgroup/garage/pull/796))
  * pycma to 2.7.0 ([#861](https://github.com/rlworkgroup/garage/pull/861))
  * tensorflow to 1.15 ([#953](https://github.com/rlworkgroup/garage/pull/953))
  * pytorch to 1.3.0 ([#952](https://github.com/rlworkgroup/garage/pull/952))

### Removed
- `garage.misc.autoargs`, a tool for decorating classes with autogenerated
  command-line arguments ([#573](https://github.com/rlworkgroup/garage/pull/573))
- `garage.misc.ext`, a module with several unrelated utilities ([#578](https://github.com/rlworkgroup/garage/pull/578))
- `config_personal.py` module, replaced by environment variables where relevant ([#578](https://github.com/rlworkgroup/garage/pull/578), [#747](https://github.com/rlworkgroup/garage/pull/747))
- `contrib.rllab_hyperopt`, an experimental module for using `hyperopt` to tune
  hyperparameters ([#684](https://github.com/rlworkgroup/garage/pull/684))
- `contrib.bichenchao`, a module of example launchers ([#683](https://github.com/rlworkgroup/garage/pull/683))
- `contrib.alexbeloi`, a module with an importance-sampling sampler and examples
  (there were merged into garage) ([#717](https://github.com/rlworkgroup/garage/pull/717))
- EC2 cluster documentation and examples ([#835](https://github.com/rlworkgroup/garage/pull/835))
- `DeterministicMLPPolicy`, because it duplicated `ContinuousMLPPolicy` ([#929](https://github.com/rlworkgroup/garage/pull/929))
- `garage.tf.layers`, a custom high-level neural network definition API, was replaced by `garage.tf.models` ([#939](https://github.com/rlworkgroup/garage/pull/939))
- `Parameterized`, which was replaced by `garage.tf.Model` ([#942](https://github.com/rlworkgroup/garage/pull/942))
- `garage.misc.overrides`, whose features are no longer needed due proper ABC
  support in Python 3 and sphinx-autodoc ([#974](https://github.com/rlworkgroup/garage/pull/942))
- `Serializable`, which became a maintainability burden and has now been
  replaced by regular pickle protocol (`__getstate__`/`__setstate__`)
  implementations, where necessary ([#982](https://github.com/rlworkgroup/garage/pull/982))
- `garage.misc.special`, a library of mostly-unused math subroutines ([#986](https://github.com/rlworkgroup/garage/pull/986))
- `garage.envs.util`, superceded by features in [akro](https://github.com/rlworkgroup/akro) ([#986](https://github.com/rlworkgroup/garage/pull/986))
- `garage.misc.console`, a library of mostly-unused helper functions for writing
  shell scripts ([#988](https://github.com/rlworkgroup/garage/pull/988))

### Fixed
- Bug in `ReplayBuffer` [#554](https://github.com/rlworkgroup/garage/pull/554)
- Bug in `setup_linux.sh` [#560](https://github.com/rlworkgroup/garage/pull/560)
- Bug in `examples/sim_policy.py` ([#691](https://github.com/rlworkgroup/garage/pull/691))
- Bug in `FiniteDifferenceHvp` ([#745](https://github.com/rlworkgroup/garage/pull/745))
- Determinism bug for some samplers ([#880](https://github.com/rlworkgroup/garage/pull/880))
- `use_gpu` in the experiment runner ([#918](https://github.com/rlworkgroup/garage/pull/918))


## [2019.02.2](https://github.com/rlworkgroup/garage/releases/tag/v2019.02.2)

### Fixed
- Bug in entropy regularization in TensorFlow PPO/TRPO ([#579](https://github.com/rlworkgroup/garage/pull/579))
- Bug in which advantage normalization was broken for recurrent policies ([#626](https://github.com/rlworkgroup/garage/pull/626))
- Bug in `examples/sim_policy.py` ([#691](https://github.com/rlworkgroup/garage/pull/691))
- Bug in `FiniteDifferenceHvp` ([#745](https://github.com/rlworkgroup/garage/pull/745))


## [2019.02.1](https://github.com/rlworkgroup/garage/releases/tag/v2019.02.1)
### Fixed
- Fix overhead in GaussianMLPRegressor by optionally creating assign operations ([#622](https://github.com/rlworkgroup/garage/pull/622))


## [2019.02.0](https://github.com/rlworkgroup/garage/releases/tag/v2019.02.0)

### Added
- Epsilon-greedy exploration strategy, DiscreteMLPModel, and
  QFunctionDerivedPolicy (all needed by DQN)
- Base Model class for TensorFlow-based primitives
- Dump plots generated with matplotlib to TensorBoard
- Relative Entropy Policy Search (REPS) algorithm
- GaussianConvBaseline and GaussianConvRegressor primitives
- New Dockerfiles, docker-compose files, and Makefiles for running garage using
  Docker
- Vanilla policy gradient loss to NPO
- Truncated Natural Policy Gradient (TNPG) algorithm for TensorFlow
- Episodic Reward Weighted Regression (ERWR) algorithm for TensorFlow
- gym.Env wrappers used for pixel environments
- Convolutional Neural Network primitive

### Changed
- Move dependencies from environment.yml to setup.py
- Update dependencies:
  - tensorflow-probability to 0.5.x
  - dm_control to commit 92f9913
  - TensorFlow to 1.12
  - MuJoCo to 2.0
  - gym to 0.10.11
- Move dm_control tests into the unit test tree
- Use GitHub standard .gitignore
- Improve the implementation of RandomizedEnv (Dynamics Randomization)
- Decouple TensorBoard from the logger
- Move files from garage/misc/instrument to garage/experiment
- setup.py to be canonical in format and use automatic versioning

### Removed
- Move some garage subpackages into their own repositories:
  - garage.viskit to [rlworkgroup/viskit](https://github.com/rlworkgroup/viskit)
  - garage.spaces to [rlworkgroup/akro](https://github.com/rlworkgroup/akro)
- Remove Theano backend, algorithms, and dependencies
- Custom environments which duplicated [openai/gym](https://github.com/openai/gym)
- Some dead files from garage/misc (meta.py and viewer2d.py)
- Remove all code coverage tracking providers except CodeCov

### Fixed
- Clean up warnings in the test suite
- Pickling bug in GaussianMLPolicyWithModel
- Namescope in LbfgsOptimizer
- Correctly sample paths in OffPolicyVectorizedSampler
- Implementation bugs in tf/VPG
- Bug when importing Box
- Bug in test_benchmark_her

## [2018.10.1](https://github.com/rlworkgroup/garage/releases/tag/v2018.10.1)

### Fixed
- Avoid importing Theano when using the TensorFlow branch
- Avoid importing MuJoCo when not required
- Implementation bugs in tf/VPG
- Bug when importing Box
- Bug in test_benchmark_her
- Bug in the CI scripts which produced false positives

## [2018.10.0](https://github.com/rlworkgroup/garage/releases/tag/v2018.10.1)

### Added
- PPO and DDPG for the TensorFlow branch
- HER for DDPG
- Recurrent Neural Network policy support for NPO, PPO and TRPO
- Base class for ReplayBuffer, and two implementations: SimpleReplayBuffer
  and HerReplayBuffer
- Sampler classes OffPolicyVectorizedSampler and OnPolicyVectorizedSampler
- Base class for offline policies OffPolicyRLAlgorithm
- Benchmark tests for TRPO, PPO and DDPG to compare their performance with
  those produced by OpenAI Baselines
- Dynamics randomization for MuJoCo environments
- Support for dm_control environments
- DictSpace support for garage environments
- PEP8 checks enforced in the codebase
- Support for Python imports: maintain correct ordering and remove unused
  imports or import errors
- Test on TravisCI using Docker images for managing dependencies
- Testing code reorganized
- Code Coverage measurement with codecov
- Pre-commit hooks to enforce PEP8 and to verify imports and commit messages,
  which are also applied in the Travis CI verification
- Docstring verification for added files that are not in the test branch or
  moved files
- TensorBoard support for all key-value/log_tabular calls, plus support for
  logging distributions
- Variable and name scope for symbolic operations in TensorFlow
- Top-level base Space class for garage
- Asynchronous plotting for Theano and Tensorflow
- GPU support for Theano

### Changed
- Rename rllab to garage, including all the rllab references in the packages
  and modules inside the project
- Rename run_experiment_lite to run_experiment
- The file cma_es_lib.py was replaced by the pycma library available on PyPI
- Move the contrib package to garage.contrib
- Move Theano-dependent code to garage.theano
- Move all code from sandbox.rocky.tf to garage.tf
- Update several dependencies, mainly:
  - Python to 3.6.6
  - TensorFlow to 1.9
  - Theano to 1.0.2
  - mujoco-py to 1.50.1
  - gym to 0.10.8
- Transfer various dependencies from conda to pip
- Separate example script files in the Theano and TensorFlow branch
- Update LICENSE, CONTRIBUTING.md and .gitignore
- Use convenience imports, that is, import classes and functions that share the
  same or similar name to its module in the corresponding `__init__.py` file of
  their package
- Replace ProxyEnv with gym.Wrapper
- Update installation scripts for Linux and macOS

### Removed
- All unused imports in the Python files
- Unused packages from environment.yml
- The files under rllab.mujoco_py were removed to use the pip release instead
- Empty `__init__.py` files
- The environment class defined by rllab.envs.Env was not imported to garage
  and the environment defined by gym.Env is used now

### Fixed
- Sleeping processes produced by the parallel sampler. NOTE: although the
  frequency of this issue has been reduced, our tests in TravisCI occasionally
  detect the issue and currently it seems to be an issue with re-entrant locks
  and multiprocessing in Python.
