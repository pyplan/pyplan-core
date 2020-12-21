import json

from pyplan_core.classes.evaluators.BaseEvaluator import BaseEvaluator


class IPythonEvaluator(BaseEvaluator):
    """ IPython Evaluator. 
    Display an IPython component using _repr_*_ methods
    """

    def evaluateNode(self, result, nodeDic, nodeId, dims=None, rows=None, columns=None, summaryBy="sum", bottomTotal=False, rightTotal=False, fromRow=0, toRow=0, hideEmpty=None, rowOrder='original', columnOrder='original'):
        return self.toHTML(result)

    def previewNode(self, nodeDic, nodeId):
        res = {
            "resultType": str(type(nodeDic[nodeId].result)),
            "dims": [],
            "columns": [],
            "console": nodeDic[nodeId].lastEvaluationConsole,
            "preview": self.toHTML(nodeDic[nodeId].result)
        }
        return json.dumps(res)

    def toHTML(self, result):
        try:
            return result._repr_html_()
        except:
            try:
                 return result._repr_pretty_()
            except:
                try:
                    return result._repr_png_()
                except:
                    try:
                        return result._repr_jpeg_()
                    except:
                        try:
                            return result._repr_json_()
                        except:
                            try:
                                return str(result)
                            except:
                                return "Result cannot be displayed"
