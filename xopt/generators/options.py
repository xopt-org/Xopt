from typing import List

from pydantic import BaseModel


class GeneratorOptions(BaseModel):
    """
    Options for the generator.
    """

    # The name of the generator.
    name: str = "xopt"

    # The version of the generator.
    version: str = "0.0.1"

    all_keys: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.check_for_duplicate_keys()

    def update(self, **kwargs):
        """
        Recursively update the options.
        """
        all_kwargs = kwargs

        def set_recursive(d: BaseModel):
            if not isinstance(d, dict):
                for name, val in d.__fields__.items():
                    attr = getattr(d, name)
                    if isinstance(attr, BaseModel):
                        set_recursive(attr)
                    elif name in kwargs.keys():
                        setattr(d, name, all_kwargs.pop(name))
                    else:
                        pass

        set_recursive(self)

        if len(all_kwargs):
            raise RuntimeError(
                f"keys {list(all_kwargs.keys())} not found, will not be " f"updated!"
            )

    def check_for_duplicate_keys(self):
        """
        Check if there are duplicate keys. If so, raise an error.
        """
        keys = self.get_all_keys()

        def check_duplicates(listOfElems):
            """Check if given list contains any duplicates"""
            setOfElems = set()
            for elem in listOfElems:
                if elem in setOfElems:
                    return True
                else:
                    setOfElems.add(elem)
            return False

        if check_duplicates(keys):
            raise RuntimeError(f"Duplicate keys found: invalid generator options!")

    def get_all_keys(self):
        """
        recursively get all keys of sub- and main-models.
        """
        keys = []
        for name, val in self.__fields__.items():
            attr = getattr(self, name)
            if isinstance(attr, GeneratorOptions):
                keys += attr.get_all_keys()
            else:
                keys += [name]

        return keys
