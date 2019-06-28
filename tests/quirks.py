"""Documented breakages and quirks caused by dependencies."""

# openai/gym environments known to not implement render()
#
# e.g.
# > gym/core.py", line 111, in render
# >     raise NotImplementedError
# > NotImplementedError
#
# Tests calling render() on these should verify they raise NotImplementedError
# ```
# with pytest.raises(NotImplementedError):
#     env.render()
# ```
KNOWN_GYM_RENDER_NOT_IMPLEMENTED = [
    # Please keep alphabetized
    'Blackjack-v0',
    'GuessingGame-v0',
    'HotterColder-v0',
    'NChain-v0',
    'Roulette-v0',
]
