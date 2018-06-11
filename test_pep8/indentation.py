"""
Identation is nicely verified by pylint, and this file provides examples
of format errors based on PEP8 that can be detected by pylint.

Test as:
pylint test_pep8/indentation.py --rcfile=config_pylint
"""


# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)


# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)


# Hanging indents should add a level.
foo = long_function_name(
      var_one, var_two,
      var_three, var_four)


def some_function_that_takes_arguments(*args):
    print(args)


"""
The closing brace/bracket/parenthesis on multiline constructs may either
line up under the first non-whitespace character of the last line of list,
as in:

my_list = [
    1, 2, 3,
    4, 5, 6,
    ]

result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
    )

or it may be lined up under the first character of the line that starts the
multiline construct, as in:

my_list = [
    1, 2, 3,
    4, 5, 6,
]

result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
)
Following statements do not follow the format, so they have to be detected
as errors.
"""

my_list = [
    1, 2, 3,
    4, 5, 6,
	]

result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
		)
