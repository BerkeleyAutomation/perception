def closure(exc):
    """
    Taken from Trimesh:
    https://github.com/mikedh/trimesh/blob/master/trimesh/exceptions.py.
    Return a function which will accept any arguments
    but raise the exception when called.
    Parameters
    ------------
    exc : Exception
      Will be raised later
    Returns
    -------------
    failed : function
      When called will raise `exc`
    """
    # scoping will save exception
    def failed(*args, **kwargs):
        raise exc

    return failed


class SensorUnresponsiveException(Exception):
    def __init__(self, *args, **kwargs):
        super(SensorUnresponsiveException, self).__init__(*args, **kwargs)
