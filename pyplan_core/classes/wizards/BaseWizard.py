import json
import autopep8
from pandas import MultiIndex, DataFrame
from xarray import DataArray


class BaseWizard(object):

    def getLastDefinition(self, definition, variable="_df"):
        newDef = definition.replace(
            "result=", f"{variable} =").replace("result =", f"{variable} =")
        return newDef

    def getColumnList(self, model, params):
        res = []
        nodeId = params["nodeId"]
        if model.existNode(nodeId):
            nodeResult = model.getNode(nodeId).result
            if isinstance(nodeResult, DataFrame):
                sample_df = nodeResult.sample(
                    min(len(nodeResult), 10000)).reset_index()
                count_by_columns = None
                if "includeColumnList" in params and params["includeColumnList"]:
                    only_string = sample_df.select_dtypes(include="object")
                    serie_count = only_string.nunique()
                    serie_count = serie_count[serie_count < 500]
                    if len(serie_count) > 0:
                        count_by_columns = serie_count

                # append columns
                for nn, col in enumerate(list(sample_df.columns)):
                    if not col is None and not (col == "index" and not nodeResult.index is None and nodeResult.index.name is None):
                        item = dict(field=col, type="index" if col in nodeResult.index.names else "column",  dtype=self.kindToString(
                            sample_df.dtypes[nn].kind))
                        if not count_by_columns is None and col in count_by_columns:
                            item["values"] = sample_df[col].dropna(
                            ).sort_values().unique().tolist()
                        res.append(item)
        return res

    def getDataArrayDimensions(self, model, params):
        res = []
        nodeId = params["nodeId"]
        if model.existNode(nodeId):
            nodeResult = model.getNode(nodeId).result
            if isinstance(nodeResult, DataArray):
                for dim in nodeResult.dims:
                    dim_node = model.getNode(
                        dim) if model.existNode(dim) else None
                    title = dim_node.title if dim_node and dim_node.title else dim
                    item = dict(field=dim, title=title,
                                values=nodeResult.coords[dim].values.tolist())
                    res.append(item)
        return res

    def kindToString(self, kind):
        """Returns the data type on human-readable string
        """
        if kind in {'U', 'S'}:
            return "string"
        elif kind in {'b'}:
            return "boolean"
        elif kind in {'i', 'u', 'f', 'c'}:
            return "numeric"
        elif kind in {'m', 'M'}:
            return "date"
        elif kind in {'O'}:
            return "object"
        elif kind in {'V'}:
            return "void"

    def formatDefinition(self, definition):
        return autopep8.fix_code(definition)
