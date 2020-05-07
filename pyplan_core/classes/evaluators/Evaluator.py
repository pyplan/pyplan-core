import numpy as np
import pandas as pd
import xarray as xr
from pyplan_core import cubepy 
from pyplan_core.classes.evaluators.BaseEvaluator import BaseEvaluator
from pyplan_core.classes.evaluators.CubepyEvaluator import CubepyEvaluator
from pyplan_core.classes.evaluators.NumpyEvaluator import NumpyEvaluator
from pyplan_core.classes.evaluators.PandasEvaluator import PandasEvaluator
from pyplan_core.classes.evaluators.XArrayEvaluator import XArrayEvaluator
import inspect


class Evaluator(object):
    ipytonMethods = ["_repr_html_", "_repr_json_",
                     "_repr_jpeg_", "_repr_png_", "_repr_pretty_"]

    @staticmethod
    def createInstance(result):
        if result is None:
            return BaseEvaluator()
        else:
            if Evaluator.isPandas(result):
                return PandasEvaluator()
            elif Evaluator.isXArray(result):
                return XArrayEvaluator()
            elif Evaluator.isMatplotlib(result):
                from pyplan_core.classes.evaluators.MatplotlibEvaluator import MatplotlibEvaluator
                return MatplotlibEvaluator()
            elif Evaluator.isNumpy(result):
                return NumpyEvaluator()
            elif Evaluator.isBokeh(result):
                from pyplan_core.classes.evaluators.BokehEvaluator import BokehEvaluator
                return BokehEvaluator()
            elif Evaluator.isPlotly(result):
                from pyplan_core.classes.evaluators.PlotlyEvaluator import PlotlyEvaluator
                return PlotlyEvaluator()
            elif Evaluator.isCubepy(result):
                return CubepyEvaluator()
            elif Evaluator.isIPython(result):
                from pyplan_core.classes.evaluators.IPythonEvaluator import IPythonEvaluator
                return IPythonEvaluator()
            else:
                return BaseEvaluator()

    @staticmethod
    def isPandas(result):
        return isinstance(result, pd.DataFrame) or isinstance(result, pd.Series) or isinstance(result, pd.Index)

    @staticmethod
    def isXArray(result):
        return isinstance(result, xr.DataArray)

    @staticmethod
    def isMatplotlib(result):
        try:
            from matplotlib.artist import Artist as MatplotlibArtist
            return isinstance(result, MatplotlibArtist) or inspect.ismodule(result) and "matplotlib.pyplot" in str(result) or isinstance(result, np.ndarray) and result.ndim > 0 and len(result) > 0 and isinstance(result.item(0), MatplotlibArtist)
        except:
            return False

    @staticmethod
    def isNumpy(result):
        return isinstance(result, np.ndarray)

    @staticmethod
    def isBokeh(result):
        try:
            from bokeh.plotting import Figure
            from bokeh.layouts import LayoutDOM
            return isinstance(result, Figure) or isinstance(result, LayoutDOM)
        except:
            return False

    @staticmethod
    def isPlotly(result):
        try:
            from plotly.graph_objs._figure import Figure as PlotlyFigue
            return isinstance(result, PlotlyFigue)
        except:
            return False

    @staticmethod
    def isCubepy(result):
        return isinstance(result, cubepy.Cube) or isinstance(result, cubepy.Index)

    @staticmethod
    def isIPython(result):
        _dir = dir(result)
        return len(list(set(_dir) & set(Evaluator.ipytonMethods))) > 0
