from pyplan_core.classes.wizards.BaseWizard import BaseWizard
import pandas as pd
import jsonpickle


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "DataframeIndex"

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]
        if model.existNode(nodeId):
            currentDef = model.getNode(nodeId).definition
            newDef = self.getLastDefinition(currentDef)
            newDef = newDef + "\n# Set index"
            
            reset_index = ""
            df = model.getNode(nodeId).result
            if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:  # df can be indexed by one column and it will not be a MultiIndex
                reset_index = ".reset_index()"

            if not params is None and "columns" in params and len(params["columns"]) > 0:
                newDef = newDef + \
                    f"\nresult = _df{reset_index}.set_index({params['columns']})"
            else:
                newDef = newDef + f"\nresult = _df{reset_index}"

            model.getNode(nodeId).definition = self.formatDefinition(newDef)
            return "ok"
        return ""
