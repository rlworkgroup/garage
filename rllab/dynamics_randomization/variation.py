class Variation:
    def __init__(self, xpath, attrib, method, distribution, var_range):
        """
        Parameters
            - xpath: path expression to identify a node within the XML file
            of the MuJoCo environment.
            - attrib: name of the dynamic parameter to randomize within the
            node defined in xpath.
            - method: if equal to "absolute", it sets the dyanmic parameter
            equal to the random coefficient obtained from the distribution, or
            if equal to "coefficient", it multiplies the default value provieded
            in the XML file by the random coefficient.
            - distribution: it specifies the probability distribution used to
            obtain the random coefficient.
            - var_range: it defines the range of values the random coefficient
            could take.
        """
        self._xpath = xpath
        self._attrib = attrib
        self._method = method
        self._distribution = distribution
        self._var_range = var_range
        self._elem = None
        self._default = None

    @property
    def xpath(self):
        return self._xpath

    @property
    def elem(self):
        return self._elem

    @elem.setter
    def elem(self, elem):
        self._elem = elem

    @property
    def attrib(self):
        return self._attrib

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, default):
        self._default = default

    @property
    def method(self):
        return self._method

    @property
    def distribution(self):
        return self._distribution

    @property
    def var_range(self):
        return self._var_range

class VariationsList:
    def __init__(self):
        self._list = []

    def add_variation(self, xpath, attrib, method, distribution, var_range):
        variation = Variation(xpath, attrib, method, distribution, var_range)
        self._list.append(variation)
        return self

    def get_list(self):
        return self._list
