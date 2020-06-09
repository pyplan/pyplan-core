from pyplan_core.classes.wizards.BaseWizard import BaseWizard
import xarray as xr
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "DataArrayFilter"

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]
        if model.existNode(nodeId) and "filters" in params:
            nodeResult = model.getNode(nodeId).result
            current_def = model.getNode(nodeId).definition
            new_def = current_def
            filter_dict = params["filters"]

            new_def = self.getLastDefinition(current_def, "_da")
            new_def = f"{new_def}\n# applied filters"
            new_def = f"{new_def}\nresult = _da.sel({json.dumps(filter_dict)}, drop=True)"

            model.getNode(nodeId).definition = self.formatDefinition(new_def)
            return nodeId
        return ""
