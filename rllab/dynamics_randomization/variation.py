from enum import Enum


class VariationMethods(Enum):
    COEFFICIENT = 1
    ABSOLUTE = 2


class VariationDistributions(Enum):
    GAUSSIAN = 1
    UNIFORM = 2


class Variation:
    def __init__(self):
        self._xpath = None
        self._attrib = None
        self._method = None
        self._distribution = None
        self._var_range = None
        self._elem = None
        self._default = None

    @property
    def xpath(self):
        return self._xpath

    @xpath.setter
    def xpath(self, xpath):
        self._xpath = xpath

    @property
    def elem(self):
        return self._elem

    @elem.setter
    def elem(self, elem):
        self._elem = elem

    @property
    def attrib(self):
        return self._attrib

    @attrib.setter
    def attrib(self, attrib):
        self._attrib = attrib

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, default):
        self._default = default

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        self._method = method

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution

    @property
    def var_range(self):
        return self._var_range

    @var_range.setter
    def var_range(self, var_range):
        self._var_range = var_range


class Variations:
    def __init__(self):
        self._list = []

    def randomize(self):
        variation = Variation()
        self._list.append(variation)
        return self

    def at_xpath(self, xpath):
        """
        Parameters
            - xpath: path expression to identify a node within the XML file
            of the MuJoCo environment.
        """
        if self._list:
            self._list[-1].xpath = xpath
        return self

    def attribute(self, attrib):
        """
        Parameters
            - attrib: name of the dynamic parameter to randomize within the
            node defined in xpath.
        """
        if self._list:
            self._list[-1].attrib = attrib
        return self

    def with_method(self, method):
        """
        Parameters
            - method: if equal to "absolute", it sets the dyanmic parameter
            equal to the random coefficient obtained from the distribution, or
            if equal to "coefficient", it multiplies the default value provieded
            in the XML file by the random coefficient.
        """
        if self._list:
            self._list[-1].method = method
        return self

    def sampled_from(self, distribution):
        """
        Parameters
            - distribution: it specifies the probability distribution used to
            obtain the random coefficient.
        """
        if self._list:
            self._list[-1].distribution = distribution
        return self

    def with_range(self, low, high):
        """
        Parameters
            - low: inclusive low value of the range
            - high: exclusive high value of the range
        """
        if self._list:
            self._list[-1].var_range = (low, high)
        return self

    def get_list(self):
        return self._list
