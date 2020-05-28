from pyplan_core.classes.wizards.BaseWizard import BaseWizard
import pandas as pd
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "DataarrayFromPandas"

    def getDataframeSchema(self, model, params):
        res = dict()
        res["columns"] = self.getColumnList(model, params)
        res["model_indexes"] = [{"title": node.title, "id": node.identifier}
                                for node in model.findNodes('nodeClass', "index")]
        res["model_indexes"].sort(key=lambda x: x["id"])
        return res

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]

        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)
            index_list = []
            domain_dic = dict()
            agg_dic = dict()
            new_def = "# Preparing the dataframe to be converted to dataarray\n"
            value_colum = ""
            previous_step = nodeId

            for index_dic in params["indexes"]:
                if not index_dic["model_index"]:
                    column_name = index_dic["field"]
                    new_index_id = model._removeDiacritics(column_name)
                    if model.existNode(new_index_id):
                        new_index_id = model.getNextIdentifier(new_index_id)

                    node_position = model.getAPlace(base_node.moduleId, int(
                        base_node.x) + 150, int(base_node.y) - 70, 96, 48)
                    new_index_node = model.createNode(
                        new_index_id, nodeClass="index", moduleId=base_node.moduleId, x=node_position["x"], y=node_position["y"])
                    new_index_node.title = column_name
                    new_index_node.definition = f"result = pd.Index({nodeId}.reset_index()['{column_name}'].unique().tolist())"
                    index_dic["model_index"] = new_index_node.identifier
                index_list.append(index_dic["field"])
                domain_dic[index_dic["field"]
                           ] = f"_#_{index_dic['model_index']}_#_"

            agg_item = params["agg"]
            column_name = agg_item["field"]
            calc = agg_item["calc"]
            value_colum = column_name
            new_node_title = f"{calc.capitalize()} of {column_name}"
            agg_dic[column_name] = calc

            # create dataarray node
            new_node_id = model._removeDiacritics(new_node_title)
            if model.existNode(new_node_id):
                new_node_id = model.getNextIdentifier(new_node_id)

            node_position = model.getAPlace(base_node.moduleId, int(
                base_node.x) + 150, int(base_node.y), 96, 48)
            new_node = model.createNode(new_node_id, nodeClass="variable",
                                        moduleId=base_node.moduleId, x=node_position["x"], y=node_position["y"])
            new_node.title = new_node_title
            domain_dic_str = json.dumps(domain_dic).replace(
                "\"_#_", "").replace("_#_\"", "")

            new_def += f"_df = {previous_step}.reset_index().groupby({json.dumps(index_list)}, as_index=False).agg({agg_dic})\n"
            new_def += f"result = pp.dataarray_from_pandas(_df, {domain_dic_str}, '{value_colum}')"
            new_node.definition = new_def

            return new_node.identifier

        return ""
