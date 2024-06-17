from environs import Env
from beartype import beartype
from beartype.door import is_bearable

# environment

env = Env()
env.read_env()

# function

def always(value):
    def inner(*args, **kwargs):
        return value
    return inner

def identity(t):
    return t

should_typecheck = env.bool('TYPECHECK', False)

typecheck = beartype if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

__all__ = [
    typecheck,
    beartype_isinstance
]
