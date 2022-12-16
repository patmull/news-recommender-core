# custom decorator
import functools
import inspect

from funcy.compat import basestring


def accepts_first_argument(*types):
    """
    Decorator for checking the accept type of the first argument.
    Sometimes behaved unpredictable when used on more than one argument,
    so should be used carefully.

    @param types:
    @return:
    """
    def check_accepts_first_argument(f):
        # This fails if other data types are not specified

        def new_f(*args, **kwds):
            # checks only for first argument
            first_type = types[0]
            checked_argument = args[1]
            # noinspection # noqa
            print("args:")
            print(args)
            print("first_type:")
            print(first_type)
            # noinspection PyTypeHints
            if not isinstance(checked_argument, first_type):
                raise ValueError("arg %r does not match %s" % (checked_argument, first_type))
            if checked_argument == "":
                raise ValueError("arg %r is empty string %s" % (checked_argument, first_type))
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts_first_argument


# TODO: This does not work as intendent when is used e.g. with SVM -- it shows data type
#  of class (well... object) itself for some reason
#  Try to finish this.
def accepts_third_argument(*types):
    @PendingDeprecationWarning
    def check_accepts_third_argument(f):
        # This fails if other data types are not specified

        def new_f(*args, **kwds):
            # checks only for first argument
            first_type = types[0]
            print("args")
            print(args)
            print("types")
            print(types)
            checked_argument = args[2]
            # noinspection PyTypeHints
            if not isinstance(checked_argument, first_type):
                raise ValueError("arg %r does not match %s" % (checked_argument, first_type))
            if checked_argument == "":
                raise ValueError("arg %r is empty string %s" % (checked_argument, first_type))
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return NotImplementedError


@PendingDeprecationWarning
def accepts_types(*types):
    def check_accepts(f):
        # This fails if other data types are not specified
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            # checks only for first argument
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)

            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def check_empty_string(f):
    """
    Empty string parameter decorator

    @param f:
    @return:
    """
    @functools.wraps(f)
    def wrapper(*a, **k):
        d = inspect.getcallargs(f, *a, **k)
        check_empty_string(d)
        return f(*a, **k)

    return wrapper


@PendingDeprecationWarning
def checking_empty_string(d):
    for name, value in d.iteritems():
        check_attribute(name, value)


def check_attribute(name, value):

    """
    Gives warnings on stderr if the value is an empty or whitespace input_string.
    All other values, including None, are OK and give no warning.
    """

    if isinstance(value, basestring) and (not value or value.isspace()):
        raise ValueError("Invalid value %r for argument %r" % (value, name))
