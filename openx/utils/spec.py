import importlib
from functools import partial
from typing import Any, Dict, Optional, Tuple, TypedDict, Union


class ModuleSpec(TypedDict):
    """A JSON-serializable representation of a function or class with some default args and kwargs to pass to
    it. Useful for specifying a particular class or function in a config file, while keeping it serializable
    and overridable from the command line using ml_collections.

    Taken from Octo https://github.com/octo-models/octo/blob/main/octo/utils/spec.py

    Usage:

        # Preferred way to create a spec:
        >>> from octo.model.components.transformer import Transformer
        >>> spec = ModuleSpec.create(Transformer, num_layers=3)
        # Same as above using the fully qualified import string:
        >>> spec = ModuleSpec.create("octo.model.components.transformer:Transformer", num_layers=3)

        # Usage:
        >>> ModuleSpec.instantiate(spec) == partial(Transformer, num_layers=3)
        # can pass additional kwargs at instantiation time
        >>> transformer = ModuleSpec.instantiate(spec, num_heads=8)

    Note: ModuleSpec is just an alias for a dictionary (that is strongly typed), not a real class. So from
    your code's perspective, it is just a dictionary.

    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @staticmethod
    def create(callable_or_full_name: Union[str, callable], *args, **kwargs) -> "ModuleSpec":  # type: ignore
        """Create a module spec from a callable or import string.

        Args:
            callable_or_full_name (str or object): Either the object itself or a fully qualified import string
                (e.g. "octo.model.components.transformer:Transformer")
        args (tuple, optional): Passed into callable upon instantiation.
        kwargs (dict, optional): Passed into callable upon instantiation.
        """
        if isinstance(callable_or_full_name, str):
            assert callable_or_full_name.count(":") == 1, (
                "If passing in a string, it must be a fully qualified import string "
                "(e.g. 'octo.model.components.transformer:Transformer')"
            )
            module, name = callable_or_full_name.split(":")
        else:
            module, name = _infer_full_name(callable_or_full_name)

        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)

    @staticmethod
    def instantiate(spec: Optional["ModuleSpec"]):  # type: ignore
        if spec is None:
            return None
        if set(spec.keys()) != {"module", "name", "args", "kwargs"}:
            raise ValueError(
                f"Expected ModuleSpec, but got {spec}. "
                "ModuleSpec must have keys 'module', 'name', 'args', and 'kwargs'."
            )
        cls = _import_from_string(spec["module"], spec["name"])
        return partial(cls, *spec["args"], **spec["kwargs"])

    @staticmethod
    def to_string(spec: "ModuleSpec"):  # type: ignore
        return (
            f"{spec['module']}:{spec['name']}"
            f"({', '.join(spec['args'])}"
            f"{', ' if spec['args'] and spec['kwargs'] else ''}"
            f"{', '.join(f'{k}={v}' for k, v in spec['kwargs'].items())})"
        )

    @staticmethod
    def is_module_spec(d: Dict):
        return isinstance(d, dict) and all([k in d for k in ("module", "name", "args", "kwargs")]) and len(d) == 4


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__name__"):
        return o.__module__, o.__name__
    raise ValueError(
        f"Could not infer identifier for {o}. "
        "Please pass in a fully qualified import string instead "
        "e.g. 'octo.model.components.transformer:Transformer'"
    )


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as e:
        raise ValueError(
            f"Could not import {module_string}:{name}."
            "If it is a partial, try using a string formatted as `module:name`"
        ) from e


def add_kwarg(d: Dict, k: str, v: Any):
    key_parts = k.split(".")
    for key_part in key_parts[:-1]:
        if ModuleSpec.is_module_spec(d):
            d = d["kwargs"]
        d = d[key_part]
    if ModuleSpec.is_module_spec(d):
        d = d["kwargs"]
    d[key_parts[-1]] = v


def recursively_instantiate(obj: Any):
    if ModuleSpec.is_module_spec(obj):
        spec = ModuleSpec(
            module=obj["module"],
            name=obj["name"],
            args=recursively_instantiate(obj["args"]),
            kwargs=recursively_instantiate(obj["kwargs"]),
        )
        return ModuleSpec.instantiate(spec)()
    if isinstance(obj, dict):
        return {k: recursively_instantiate(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(map(recursively_instantiate, obj))
    return obj
