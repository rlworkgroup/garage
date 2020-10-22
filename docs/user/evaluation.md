# Evaluation

garage provides two useful function to evaluate the performance of an
algorithm, [log_performance](../_autoapi/garage/index.html#garage.log_performance)
and [log_multitask_performance](../_autoapi/garage/index.html#garage.log_multitask_performance).
`log_performance` is used for generous algorithms, while
`log_multitask_performance` is used for multiple tasks algorithms.

The input of the both functions is [EpisodeBatch](../_autoapi/garage/index.html#garage.EpisodeBatch),
which is a batch of episodes.

These functions will evaluate algorithms in from the following aspects:

- `AverageReturn`: The average return (sum of rewards in an episode) of all
episodes.

- `AverageDiscountedReturn`: The average discounted return of all episodes.

- `StdReturn`: The standard deviation of undiscounted returns.

- `MaxReturn`: The maximum undiscounted return.

- `MinReturn`: The minimum undiscounted return.

- `TerminationRate`: Terminated episodes / all episodes.

- `SuccessRate` (if applicable): The rate of success among all episodes.

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
