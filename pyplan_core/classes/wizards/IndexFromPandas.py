from pyplan_core.classes.wizards.BaseWizard import BaseWizard
import pandas as pd
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "IndexFromPandas"

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]
        res = ""

        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)

            for column_name in params["indexes"]:
                new_index_id = model._removeDiacritics(column_name)
                if model.existNode(new_index_id):
                    new_index_id = model.getNextIdentifier(new_index_id)

                node_position = model.getAPlace(base_node.moduleId, int(
                    base_node.x) + 150, int(base_node.y), 96, 48)
                new_index_node = model.createNode(
                    new_index_id, nodeClass="index", moduleId=base_node.moduleId, x=node_position["x"], y=node_position["y"])

                new_index_node.title = column_name
                new_index_node.definition = f"result = pd.Index({nodeId}.reset_index()['{column_name}'].unique().tolist())"
                res = new_index_node.identifier

        return res
        
