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
    """ Guassian distribution """
    GAUSSIAN = 1
    """ Uniform distribution """
    UNIFORM = 2


class Variation:
    """
    Each dynamic parameter to randomize is represented by a Variation. This
    class works more like a data structure to store the data fields required
    to find the dynamic parameter and the randomization to apply to it.
    """

    def __init__(self):
        self._xpath = None
        self._attrib = None
        self._method = None
        self._distribution = None
        self._var_range = None
        self._elem = None
        self._default = None
        self._mean_std = None

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

    @property
    def mean_std(self):
        return self._var_range

    @mean_std.setter
    def mean_std(self, mean_std):
        self._mean_std = mean_std


class VariationsBase:
    """
    The purpose of this class is to keep a list of all the variations
    that have to be applied to the RandomizedEnv class.
    The class implements the fluent interface pattern, so each call
    to set an attribute will return the instance of this class.
    """

    def __init__(self, variations_list=[]):
        self._list = variations_list

    def randomize(self):
        """
        Creates a new entry in the list of variations. After calling this
        method, call the setters for each of the attributes to be used with
        this new entry using the fluent interface pattern.
        """
        variation = Variation()
        self._list.append(variation)
        return Variations(self._list)

    def at_xpath(self, xpath):
        """
        Sets the xpath for the last variation in the list.

        Parameters
        ----------
        xpath : string
            path expression to identify a node within the XML file
            of the MuJoCo environment.
        """
        if self._list:
            self._list[-1].xpath = xpath
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
        if self._list:
            self._list[-1].attrib = attrib
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
        if self._list:
            self._list[-1].method = method
        return self

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


class Variations(VariationsBase):
    """
    Contains all the methods that have to be called once per variation entry.
    """

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
        if self._list:
            self._list[-1].distribution = distribution

        if distribution is Distribution.GAUSSIAN:
            return VariationsGaussian(self._list)
        elif distribution is Distribution.UNIFORM:
            return VariationsUniform(self._list)
        return self


class VariationsUniform(VariationsBase):
    """
    Contains all the methods for variation entries with uniform distributions
    """

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
        if self._list:
            self._list[-1].var_range = (low, high)
        return self


class VariationsGaussian(Variations):
    """
    Contains all the methods for variation entries with Gaussian distributions
    """

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
        if self._list:
            self._list[-1].mean_std = (mean, std_deviation)
        return self
