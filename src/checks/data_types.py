@DeprecationWarning
def accepts(*types):
    def check_accepts(f):
        try:
            assert len(types) == f.__code__.co_argcount
        except AssertionError as e:
            raise ValueError(e.args)

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                try:
                    assert isinstance(a, t), "arg %r does not match %s" % (a, t)
                except AssertionError as e:
                    raise ValueError(e.args)
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts