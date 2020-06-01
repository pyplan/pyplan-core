from pyplan_core.classes.wizards.BaseWizard import BaseWizard
import json


class Wizard(BaseWizard):

    def __init__(self):
        self.code = "CreateIndex"

    def generateDefinition(self, model, params):
        nodeId = params["nodeId"]
        if model.existNode(nodeId):
            base_node = model.getNode(nodeId)
            new_def = None

            if params["type"]=="list":
                index_values = params["values"]
                new_def = f"result = pd.Index({index_values})"
            elif params["type"]=="range":
                if params["rangeType"]=="string":
                    prefix = params["stringPrefix"]
                    index_values = [f'{prefix}{nn}' for nn in range(int(params["from"]), int(params["to"])+1, int(params["step"]))]
                    str_values = json.dumps(index_values)
                    new_def = f"result = pd.Index({str_values})"
                elif params["rangeType"]=="numeric":
                    index_values = list(range(int(params["from"]), int(params["to"])+1, int(params["step"])))
                    str_values = json.dumps(index_values)
                    new_def = f"result = pd.Index({str_values})"
                elif params["rangeType"]=="date":
                    freq = params["freq"]
                    start = params["start"] if "start" in params and params["start"] else None
                    end = params["end"] if "end" in params and params["end"] else None
                    periods = params["periods"] if "periods" in params and params["periods"] else None
                    if start and end:
                        new_def = f"result = pd.DatetimeIndex(freq='{freq}', start='{start}', end='{end}')"
                    elif end:
                        new_def = f"result = pd.DatetimeIndex(freq='{freq}', end='{end}', periods={periods})"
                    else:
                        new_def = f"result = pd.DatetimeIndex(freq='{freq}', start='{start}', periods={periods})"

            #test and set new definition
            if new_def:
                temp_res = model.evaluate(new_def)
                base_node.definition = new_def
            
        return nodeId
