"""
Limit all lines to maximum 79 characters
"""


"""
This is a very long documentation line to test if the format error is detected by the lint tool
"""
print("Printing a very long message to test if the format error is detected by the lint tool")

class MyTime():
    """
    MyTime
    This is a very long documentation line to test if the format error is detected by the lint tool
    """
    def __init__(self):
        """
        __init__
        """
        self._my_time = 0
    def set_mytime(self, time):
        """
        set_mytime
        """
        self._my_time = time
    def a_very_long_method_name_to_test_if_the_format_error_is_detected_by_the_lint_tool(self):
        """
        get_mytime
        """
        print("Printing a very long message to test if the format error is detected by the lint tool")
        return self._my_time
