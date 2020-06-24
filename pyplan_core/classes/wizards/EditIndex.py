from pyplan_core.classes.wizards.BaseWizard import BaseWizard
from pandas import Index
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "EditIndex"

    def getIndexItems(self, model, params):
        nodeId = params["nodeId"]
        res = []
        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)
            pd_index = base_node.result
            if isinstance(pd_index, Index):
                res = pd_index.values.tolist()
        return res

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]
        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)
            index_values = params["values"]
            index_values_str = json.dumps(index_values, ensure_ascii=False)
            base_node.definition = f"result = pd.Index({index_values_str})"
            return nodeId

        return ""
