from enum import Enum


class Method(Enum):
    """
    The random coefficient is applied according to these methods.
    """
    """ The randomization is the product of the coefficient and the dynamic parameter """
    COEFFICIENT = 1
    """ The randomization is equal to the coefficient """
    ABSOLUTE = 2


class Distribution(Enum):
    """
    The different ways to produce the random coefficient.
    """
    """ Gaussian distribution """
    GAUSSIAN = 1
    """ Uniform distribution """
    UNIFORM = 2


class Variation:
    """
    Each dynamic parameter to randomize is represented by a Variation. This
    class works more like a data structure to store the data fields required
    to find the dynamic parameter and the randomization to apply to it.
    """

    def __init__(self,
                 xpath=None,
                 attrib=None,
                 method=None,
                 distribution=None,
                 var_range=None,
                 mean_std=None,
                 elem=None,
                 default=None):

        if distribution is Distribution.GAUSSIAN and mean_std is None:
            raise ValueError(
                "Need to call with_mean_std when sampled from Gaussian")

        if distribution is Distribution.UNIFORM and var_range is None:
            raise ValueError(
                "Need to call with_range when sampled from Uniform")

        self._xpath = xpath
        self._attrib = attrib
        self._method = method
        self._distribution = distribution
        self._var_range = var_range
        self._mean_std = mean_std
        self._elem = elem
        self._default = default

    @property
    def xpath(self):
        return self._xpath

    @property
    def elem(self):
        return self._elem

    @property
    def attrib(self):
        return self._attrib

    @property
    def default(self):
        return self._default

    @property
    def method(self):
        return self._method

    @property
    def distribution(self):
        return self._distribution

    @property
    def var_range(self):
        return self._var_range

    @property
    def mean_std(self):
        return self._mean_std


class Variations:
    """
    The purpose of this class is to keep a list of all the variations
    that have to be applied to the RandomizedEnv class.
    """

    def __init__(self):
        self._list = []

    def randomize(self):
        """
        Creates a VariationSpec instance to store values of dynamic parameters.

        Returns
        -------
        VariationSpec
        """
        return VariationSpec(self)

    def get_list(self):
        """
        Returns a list with all the variations

        Returns
        -------
        [Variation]
        A list of all the dynamic parameters to find in the model XML
        and the configuration to randomize each of them
        """
        return self._list

    def add(self, variation):
        self._list.append(variation)


class VariationSpec:
    """
    The purpose of this class is to set the values of each dynamic
    parameter.
    The class implements the fluent interface pattern, so each call
    to set an attribute will return the instance of this class.
    """

    def __init__(self, variations):
        self._variations = variations
        self._xpath = None
        self._attrib = None
        self._method = None
        self._distribution = None
        self._mean_std = None
        self._var_range = None
        self._elem = None
        self._default = None

    def at_xpath(self, xpath):
        """
        Sets the xpath for the last variation in the list.

        Parameters
        ----------
        xpath : string
            path expression to identify a node within the XML file
            of the MuJoCo environment.
        """
        self._xpath = xpath
        return self

    def attribute(self, attrib):
        """
        Sets the attribute for the last variation in the list.

        Parameters
        ----------
        attrib : string
            name of the dynamic parameter to randomize within the
            node defined in xpath.
        """
        self._attrib = attrib
        return self

    def with_method(self, method):
        """
        Sets the method to apply the random coefficient for the last variation
        in the list.

        Parameters
        ----------
        method : Method
            if equal to "absolute", it sets the dynamic parameter
            equal to the random coefficient obtained from the distribution, or
            if equal to "coefficient", it multiplies the default value provided
            in the XML file by the random coefficient.
        """
        self._method = method
        return self

    def sampled_from(self, distribution):
        """
        Sets the distribution where the random coefficient is sampled from for
        the last variation in the list.

        Parameters
        ----------
        distribution : Distribution
            it specifies the probability distribution used to obtain the random
            coefficient.
        """
        self._distribution = distribution
        return self

    def with_mean_std(self, mean, std_deviation):
        """
        Sets the range for the random coefficient for the last variation in
        the list. Only to be used for Distribution.GAUSSIAN

        Parameters
        ----------
        mean : int
           mean of the distribution
        std_deviation : int
           standard mean of the distribution
        """
        self._mean_std = (mean, std_deviation)
        return self

    def with_range(self, low, high):
        """
        Sets the range for the random coefficient for the last variation in
        the list. Only to be used for Distribution.UNIFORM

        Parameters
        ----------
        low : int
            inclusive low value of the range
        high : int
            exclusive high value of the range
        """
        self._var_range = (low, high)
        return self

    def add(self):
        self._variations.add(
            Variation(
                xpath=self._xpath,
                attrib=self._attrib,
                method=self._method,
                distribution=self._distribution,
                var_range=self._var_range,
                mean_std=self._mean_std,
                elem=self._elem,
                default=self._default))
