import operator
import inspect
import numpy as np
from collections.abc import Mapping, Iterable, Generator
from dataclasses import dataclass
from functools import wraps

class SizedIterable:
    """
    Iterable object which knows its length. Note this class is intended for use
    with iterables that don't provide their length (e.g. generators), and as
    such cannot check that the provided length is correct.

    Sourced from mackelab_toolbox.utils
    """
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length
    def __iter__(self):
        return iter(self.iterable)
    def __len__(self):
        return self.length

class LazyDict(dict):
    """
    A dictionary for lazily evaluated expressions.
    Values must be initialized with functions (typically lambda functions).
    The first time a key is accessed, the associated function is called, and the value
    in the dictionary replaced by the result. Subsequent calls will use the computed
    value. In effect this implements caching or memoization, except that the caching
    is implemented at the level of the dictionary instead of attached to each individual function.
    """
    def __init__(self, *args, **kwds):
        self._creators_dict = dict(*args, **kwds)
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            try:
                creator = self._creators_dict[key]
            except KeyError:
                raise e  # Raise the first exception, with the public facing dict
            else:
                self[key] = creator()
                return self[key]
    def __getattr__(self, attr):
        return self[attr]

def flatten(iterable) -> Generator:
    for a in iterable:
        if isinstance(a, (str, bytes, Mapping, np.ndarray)):
            yield a                # Iterable types we don’t want to expand
        elif isinstance(a, Iterable):
            yield from flatten(a)  # The rest of iterable types we expand
        else:
            yield a                # Anything non-iterable is returned as-is

def invert_dict(d) -> dict:
    """Return a new dictionary, with swapped keys and values.

    Values must be hashable, otherwise they are not valid keys.

    One-to-many relations: If a value is a list or set (which normally would not
    be a valid key), it is assumed to denote a one-to-many relation. The list/set
    is then expanded and each value used as a key. So for example

        invert_dict({operator.add: {"add", "+"}, operator.sub: ["sub", "-"]}

    becomes

        {"add": operator.add, "+": operator.add, "sub": operator.sub, "-": operator.sub}
    """
    def items(d):
        for k,v in d.items():
            if isinstance(v, (list, set)):
                for el in v:
                    yield el, k
            else:
                yield v, k

op_dict = {operator.add: {"add", "+"},
           operator.sub: {"sub", "-"},
           operator.mul: {"mul", "*"},
           operator.ge:  {"ge", ">=", "⩾"},
           operator.le:  {"le", "<=", "⩽"},
           operator.eq:  {"eq", "=", "=="},
           operator.gt:  {"gt", ">"},
           operator.lt:  {"lt", "<"}
          }
op_dict = invert_dict(op_dict)

def skip_cache(__skipcache_cmp__=None, **__skipcache_kwds__):
    """
    Conditionally deactivate a cache implemented with `functools.cache` or
    `functools.lru_cache`. Useful if caching should only be performed under
    certain circumstances (e.g. to avoid especially large result objects,
    or arguments we know will not be reused.)
    Two ways to disable the cache:

    - Using keyword args to associate tests to specific argument values.
      Multiple keyword args are combined with 'and', so the following will
      skip the cache only if both `L⩾1000` and `dist=="Cauchy"`.

          @skip_cache(L=(">=", 1000), dist=("=", "Cauchy"))
          @cache
          def draw_samples(dist:str, L:int):
            ...

      For to combine conditions with 'or', use multiple decorators::

          @skip_cache(L=(">=", 1000))
          @skip_cache(dist=("=", "Cauchy"))
          @cache
          def draw_samples(dist:str, L:int):
            ...

      LIMITATION: With these decorators, `draw_cache` can only be called with 
      keyword arguments. (Specifically, all arguments used in a `skip_cache`
      decorator must be passed by keyword.)

    - Arbitrary function: Define a test function taking all args and kwargs
      and returning True (do cache) or False (do not cache).

          def cmp(dist:str, L:int) -> bool:
            return L>=1000 and dist=="Cauchy"
          @skip_cache(cmp)
          @cache
          def draw_samples(dist:str, L:int):
            ...

    If both a comparison function and keyword args are specified, the cache
    is skipped only if all are True (i.e. they are combined with 'and').
    """
    def decorator(f):
        # If there is an error in the test specification, better detect that
        # now rather than in the middle of a long computation.
        # Ensure that `f` is a cached function
        if not hasattr(f, "__wrapped__"):
            raise RuntimeError("The `skip_cache` decorator is meant to be used with functions decorated with `functools.cache` or `functools.lru_cache`. "
                               f"This is not the case for {f}.")
        # Ensure that all 'skip_cache' arguments match a function parameter
        unrecognized_params = __skipcache_kwds__.keys() - inspect.signature(f).parameters.keys()
        if unrecognized_params:
            raise ValueError("Invalid `skip_cache` condition: The following parameters are not part "
                             f"of the function signature: {unrecognized_params}.")
        # Ensure that keyword tests are well-formed
        for test in __skipcache_kwds__.values():
            if len(test) != 2:
                raise ValueError("Tests should be tuples with two elements: (test operator, value).\n"
                                 f"Received '{test}'.")
            if test[0] not in op_dict:
                raise ValueError(f"Unrecognized test operaton '{test[0]}'. Possible values are:\n"
                                 + ", ".join(op_dict.keys()))
        @wraps(f)
        def wrapper(*args, **kwds):
            skip = False
            if __skipcache_cmp__ is not None:
                skip |= __skipcache_cmp__(*argsk, **kwargs)
            for kw, (test_op, test_val) in __skipcache_kwds__.items():
                try:
                    val = kwds[kw]
                except KeyError:
                    raise TypeError(f"{kw} argument must be passed by keyword.")
                skip |= op_dict[test_op](val, test_val)

            if skip:
                return f.__wrapped__(*args, **kwds)
            else:
                return f(*args, **kwds)

        return wrapper
    return decorator




@dataclass(frozen=True)  # Allows hashing the resulting model
class compose:
    """Compose multiple functions.

    Functions are composed RIGHT TO LEFT, so that the following are equivalent::

        f(g(h(x)))
        compose(f,g,h)(x)

    Functions must accept a single positional argument, but that argument may take any form.
    So ``f`` can return a tuple and ``g`` can return a dictionary, and it is up to ``g`` and
    ``h`` to treat them accordingly.
    
    Keyword arguments are passed to each function, so the following are equivalent::

        f(g(x, a=1), a=1)
        compose(f,g)(x, a=1)

    Keyword arguments are useful for specifying parameters and RNGs.

    Functions within the composition can be retrieved by index:

        >>> φ = compose(f,g)
        >>> φ[0] is f
        True
        >>> Φ[1] is g
        True
        >>> Φ[-1] is g
        True

    .. Note:: `compose` is implemented as a frozen dataclass. This means that if all constituant
       functions ``f``, ``g``, etc. can be hashed, than so to can the composition ``f(g(...))``.
       This allows composed functions to be used as keys in dictionaries.
    """
    funcs: tuple
    def __init__(self, *args):
        object.__setattr__(self, "funcs", args)
    def __getitem__(self, index):
        return self.funcs[index]
    def __iter__(self):
        return iter(self.funcs)
    def __call__(self, x, **kwds):
        for f in reversed(self.funcs):
            x = f(x, **kwds)
        return x

## Random number generation ##

# Project-specific mother seed: All RNGs in this project are seeded with this number
entropy = 231868418911922305076391806541830040449

# Keep track of the produced seeds. This is used to detect the unlikely
# but not impossible situation where two different entropy arguments produce
# the same seed.
produced_seeds = {}

class SeedCollisionError(RuntimeError):
    pass

def get_rng(*args: list[int|float|bytes|str]):
    """Return a pseudo-random number generator (PRNG) using the provided arguments as entropy

    Arguments are intended to be human-readable values, like parameter values or
    flags. In particular they do not need to be high-quality seeds: this
    function concatenates them with a project-unique large entropy value,
    and then uses the functionality of `numpy.random.SeedSequence` to combine
    those sources of entropy into a high-quality seed. 

    Providing the same arguments twice will produce equivalent PRNGs which return
    the same sequence of bits.

    .. Note:: There is a small possibility that two different sets of arguments
       will produce the same PRNG. For example

       >>> get_rng(1.)
       >>> get_rng(1, 1)

       both produce the same PRNG because of the way floats are internally
       converted to integers. On the other hand, collision hashes are exceedingly
       unlikely (basically the same probability as a SHA256 collision) if
       all `get_rng` calls use the same types of arguments in the same order.

       That said, even without consistent arguments, the probability is low, and
       in the case it happens, a `SeedCollisionError` (subclass of
       `RuntimeError`) is raised.
    """
    # Arguments may contain nested tuples: Flatten them
    new_args = flatten(args)
    # Convert all floats to pairs of ints  # NB: SeedSequence flattens tuples of ints
    new_args = tuple(a.as_integer_ratio() if isinstance(a, float) else a
                     for a in new_args)
    # Convert all strings to bytes
    new_args = tuple(a.encode("utf-8") if isinstance(a, str) else a
                     for a in new_args)
    # Convert all bytes to integers
    new_args = tuple(int.from_bytes(a, "little") if isinstance(a, bytes) else a
                     for a in new_args)
    # Create the seed sequence, and check if the first seed state it produces
    # collides with a previously produced one.
    seedseq = np.random.SeedSequence((entropy, *new_args))
    first_seed = seedseq.generate_state(1)[0]
    match_args = produced_seeds.get(first_seed, None)
    if match_args is None:
        produced_seeds[first_seed] = args
    elif args != match_args:
        raise SeedCollisionError(
            "The following two sets of arguments produce the same seed:\n"
            f"  {args}\n  {match_args}")

    # Return the new PRNG
    return np.random.Generator(np.random.PCG64(seedseq))

