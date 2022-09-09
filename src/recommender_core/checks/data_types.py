# custom decorator
import functools
import inspect

from funcy.compat import basestring


def accepts(*types):
    def check_accepts(f):
        assert len(types) == f.__code__.co_argcount
        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                       "arg %r does not match %s" % (a,t)
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
    Gives warnings on stderr if the value is an empty or whitespace string.
    All other values, including None, are OK and give no warning.
    """
    if isinstance(value, basestring) and (not value or value.isspace()):
        raise ValueError("Invalid value %r for argument %r" % (value, name))