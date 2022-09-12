# custom decorator
import functools
import inspect

from funcy.compat import basestring


def accepts_first_argument(*types):
    def check_accepts_first_argument(f):
        # This fails if other data types are not specified

        def new_f(*args, **kwds):
            # checks only for first argument
            first_type = types[0]
            # noinspection # noqa
            assert isinstance(args[1], first_type)
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts_first_argument


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
    @functools.wraps(f)
    def wrapper(*a, **k):
        d = inspect.getcallargs(f, *a, **k)
        check_empty_string(d)
        return f(*a, **k)

    return wrapper


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
