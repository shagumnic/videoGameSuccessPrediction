# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_tensorflow_lite_wrap_calibration_wrapper', [dirname(__file__)])
        except ImportError:
            import _tensorflow_lite_wrap_calibration_wrapper
            return _tensorflow_lite_wrap_calibration_wrapper
        if fp is not None:
            try:
                _mod = imp.load_module('_tensorflow_lite_wrap_calibration_wrapper', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _tensorflow_lite_wrap_calibration_wrapper = swig_import_helper()
    del swig_import_helper
else:
    import _tensorflow_lite_wrap_calibration_wrapper
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


class CalibrationWrapper(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CalibrationWrapper, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CalibrationWrapper, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    __swig_getmethods__["CreateWrapperCPPFromBuffer"] = lambda x: _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_CreateWrapperCPPFromBuffer
    if _newclass:
        CreateWrapperCPPFromBuffer = staticmethod(_tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_CreateWrapperCPPFromBuffer)
    __swig_destroy__ = _tensorflow_lite_wrap_calibration_wrapper.delete_CalibrationWrapper
    __del__ = lambda self: None

    def Prepare(self):
        return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_Prepare(self)

    def FeedTensor(self, input_value):
        return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_FeedTensor(self, input_value)

    def QuantizeModel(self, *args):
        return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_QuantizeModel(self, *args)

    def Calibrate(self):
        return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_Calibrate(self)
CalibrationWrapper_swigregister = _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_swigregister
CalibrationWrapper_swigregister(CalibrationWrapper)

def CalibrationWrapper_CreateWrapperCPPFromBuffer(data):
    return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_CreateWrapperCPPFromBuffer(data)
CalibrationWrapper_CreateWrapperCPPFromBuffer = _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_CreateWrapperCPPFromBuffer

# This file is compatible with both classic and new-style classes.


