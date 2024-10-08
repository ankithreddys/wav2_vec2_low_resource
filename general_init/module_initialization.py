import importlib



def initialize_module(path, args, initialize = True):
    module_path = ".".join(path.split(".")[:-1])
    class_name = path.split(".")[-1]

    module = importlib.import_module(module_path)
    Class = getattr(module, class_name)

    if initialize:
        if args:
            return Class(**args)
        else:
            return Class()
    else:
        return Class