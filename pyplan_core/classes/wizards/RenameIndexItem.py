from pyplan_core.classes.wizards.BaseWizard import BaseWizard
from pyplan_core.classes.evaluators.XArrayEvaluator import XArrayEvaluator
from pandas import Index
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "RenameIndexItem"

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
            for output_node in base_node.outputs:
                if model.existNode(output_node) and model.getNode(output_node).nodeClass == "inputtable":
                    input_table_node = model.getNode(output_node)
                    da = input_table_node.result  # to force apply previous changes
                    if nodeId in da.coords:
                        da.coords[nodeId] = index_values

                    # generate new definition
                    evaluator = XArrayEvaluator()
                    input_table_node._result = da
                    new_def = evaluator.generateNodeDefinition(
                        model.nodeDic, output_node)

                    # test and set new definition:
                    _ = model.evaluate(new_def)
                    input_table_node.definition = new_def
            base_node.definition = f"result = pd.Index({index_values_str})"
            return nodeId

        return ""
