class BadClass():
    def __init__(self):
        pass


ins = BadClass()
if not "m" in ins:  # E713 	Test for membership should be 'not in'
    pass
