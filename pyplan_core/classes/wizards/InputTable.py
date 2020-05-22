from pyplan_core.classes.wizards.BaseWizard import BaseWizard
from pyplan_core.classes.evaluators.XArrayEvaluator import XArrayEvaluator
import xarray as xr
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "InputTable"

    def getNodeSchema(self, model, params):
        nodeId = params["nodeId"]

        res = dict()
        res["model_indexes"] = [{"title": node.title if node.title else "",
                                 "id": node.identifier} for node in model.findNodes('nodeClass', "index")]
        res["model_indexes"].sort(key=lambda x: x["id"])

        res["default_value"] = "0."
        res["node_indexes"] = []
        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)
            da = base_node.result
            if isinstance(da, xr.DataArray):
                # get default value
                if "_input_properties" in base_node.definition:
                    _input_properties = model.evaluate(base_node.definition.split("# values")[
                                                       0] + "result = _input_properties")
                    res["default_value"] = str(_input_properties["defaultValue"]) if str(
                        _input_properties["defaultValue"]) else "''"

                model_indexes = res["model_indexes"]
                node_indexes = res["node_indexes"]
                for dim in da.dims:
                    node_index = next(
                        (index for index in model_indexes if index["id"] == dim), None)
                    model_indexes.remove(node_index)
                    node_indexes.append(node_index)
        return res

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]
        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)

            default_value = params["default_value"]
            if default_value:
                try:
                    default_value = int(default_value)
                except:
                    try:
                        default_value = float(default_value)
                    except:
                        if default_value in ["True", "False"]:
                            default_value = default_value == "True"
                        else:
                            default_value = default_value.strip(
                                "\"").strip("\'")

                if str(default_value).isnumeric():
                    default_value = int(default_value)

                new_indexes = params["node_indexes"]

                da = base_node.result
                if not isinstance(da, xr.DataArray):
                    da = xr.DataArray(default_value, coords=[])

                # remove unused dims
                for dim in da.dims:
                    if not dim in new_indexes:
                        da = da.isel({dim: 0}).drop(dim).squeeze()

                # add new dims
                for new_index in new_indexes:
                    if not new_index in da.dims:
                        da = da.expand_dims(
                            {new_index: list(model.getNode(new_index).result.values)})

                # generate new definition
                evaluator = XArrayEvaluator()
                base_node._result = da
                new_def = evaluator.generateNodeDefinition(
                    model.nodeDic, nodeId, defaultValue=default_value)

                # test def:
                temp_res = model.evaluate(new_def)
                if not isinstance(temp_res, xr.DataArray):
                    raise ValueError(
                        f"New definition for node is not DataArray. New type: {type(temp_res)} ")
                base_node.definition = new_def
            else:
                raise ValueError("Default value can't be empty")
            return "ok"

        return ""
