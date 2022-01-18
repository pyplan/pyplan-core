import re
import time

import numpy as np
import xarray as xr
from pyplan_core.classes.dynamics.BaseDynamic import BaseDynamic
from pyplan_core.cubepy.Helpers import Helpers


class PureXArrayDynamic(BaseDynamic):

    def circularEval(self, node, params):
        """Used for execute nodes with circular reference (pp.dynamic)"""

        dynamicVars = params["dynamicVars"]
        dynamicIndex = params["dynamicIndex"]
        nodesInCyclic = params["nodesInCyclic"]
        initialValues = params["initialValues"]
        shift = params["shift"]
        sliceInputs = params["sliceInputs"]

        evaluate = node.model.evaluate

        if node.model.debugMode:
            for circular_node_id in nodesInCyclic:
                circular_node = node.model.getNode(circular_node_id)
                if not circular_node is None:
                    node.model.sendStartCalcNode(circular_node_id, fromDynamic=node.identifier)

        # create nodes array
        cyclicNodes = []
        nodesWoDynamicIndex = []
        external_inputs = dict()
        general_time = 0

        try:
            node.model.inCyclicEvaluate = True
            for nodeId in nodesInCyclic:
                evaluate_start_time = time.time()
                _nodeObj = node.model.getNode(nodeId)
                _nodeResult = _nodeObj.bypassCircularEvaluator().result

                cyclic_item = {
                    "node": _nodeObj,
                    "initialize": self.generateInitDef(node, _nodeResult, dynamicIndex),
                    "calcTime": _nodeObj.lastEvaluationTime
                }
                cyclic_item["loopDefinition"] = self.generateLoopDef(
                    node, _nodeObj.definition, nodesInCyclic)

                cyclicNodes.append(cyclic_item)
                if isinstance(_nodeResult, xr.DataArray) and dynamicIndex.name not in _nodeResult.dims:
                    nodesWoDynamicIndex.append(_nodeObj.identifier)

                if sliceInputs:
                    # Get external inputs
                    _external_inputs_ids = set(
                        _nodeObj.inputs) - set(nodesInCyclic)
                    for node_id in _external_inputs_ids:
                        if node_id not in external_inputs and node.model.existNode(node_id):
                            input_result = node.model.getNode(node_id).result
                            if isinstance(input_result, xr.DataArray) and dynamicIndex.name in input_result.dims:
                                external_inputs[node_id] = input_result

                cyclic_item["calcTime"] += time.time() - evaluate_start_time

        except Exception as e:
            raise e
        finally:
            node.model.inCyclicEvaluate = False

        if nodesWoDynamicIndex:
            nodesWoDynamicIndexStr = ', '.join(nodesWoDynamicIndex)
            message = f"WARNING: the following nodes do not have '{dynamicIndex.name}' as a dimension: {nodesWoDynamicIndexStr}"
            consoleMessage = message
            # Set in node console
            nodeLastEvaluationConsole = node.lastEvaluationConsole
            if isinstance(nodeLastEvaluationConsole, str):
                # Concat warning with previous console message
                consoleMessage = f'{consoleMessage}\n{nodeLastEvaluationConsole}'
            node.lastEvaluationConsole = consoleMessage

        cyclicDic = {}
        # initialice var dictionary
        for _node in cyclicNodes:
            evaluate_start_time = time.time()

            _id = _node["node"].identifier

            cyclicDic[_id] = evaluate(
                _node["initialize"], returnEvaluateTime=False, fromDynamic=True)

            if not initialValues is None and _id in initialValues:
                initial_result = evaluate(
                    initialValues[_id], returnEvaluateTime=False, fromDynamic=True)
                cyclicDic[_id] = cyclicDic[_id] + initial_result
            # check align
            if cyclicDic[_id].dims[0] != dynamicIndex.name:
                # move dynamic index to top
                list_dims = list(cyclicDic[_id].dims)
                list_dims.remove(dynamicIndex.name)
                new_tuple = (dynamicIndex.name,) + tuple(list_dims)
                cyclicDic[_id] = cyclicDic[_id].transpose(
                    *new_tuple, transpose_coords=True)

            _node["calcTime"] += time.time() - evaluate_start_time

        # initialice vars in t-1
        for _var in dynamicVars:
            _key = "__" + _var + "_t"
            cyclicDic[_key] = cyclicDic[_var].sum(dynamicIndex.name) * 0

        cyclicParams = None

        theRange = range(0, len(dynamicIndex.values))
        initialCount = shift*-1
        reverseMode = False
        if shift > 0:
            theRange = range(len(dynamicIndex.values)-1, -1, -1)
            initialCount = len(dynamicIndex.values)-1-shift
            reverseMode = True

        for nn in theRange:
            item = dynamicIndex.values[nn]
            loc_dic = {dynamicIndex.name: slice(item, item)}

            if sliceInputs:
                # Overwrite external inputs result
                for external_input_id, external_input in external_inputs.items():
                    evaluate_start_time = time.time()

                    try:
                        input_without_time = external_input.loc[loc_dic].squeeze(
                            drop=True)
                        if len(input_without_time.dims) == 0:
                            input_without_time = input_without_time.item(0)

                        node.model.getNode(
                            external_input_id)._result = input_without_time
                    except Exception as ex:
                        print(
                            f'"\033[91mERROR FILTRANDO EXTERNAL INPUTS {ex}\033[0m')

                    general_time += time.time() - evaluate_start_time

            # load params
            cyclicParams = {
                "item": item,
                "cyclicDic": cyclicDic,
                "dynamicIndex": dynamicIndex,
                "self": self
            }

            # loop over variables
            for _node in cyclicNodes:
                evaluate_start_time = time.time()

                _id = _node["node"].identifier
                node.model.currentProcessingNode(_id)

                # execute vars
                if (_id in initialValues) and ((nn < initialCount and (not reverseMode)) or (nn > initialCount and reverseMode)):
                    try:
                        # use initial values
                        _resultNode = evaluate(
                            _node["loopDefinition"], cyclicParams, False, fromDynamic=True)
                        _initialValues = evaluate(
                            initialValues[_id], returnEvaluateTime=False, fromDynamic=True)
                    except Exception as ex:
                        raise ValueError(
                            f"Node '{_id}' failed during dynamic evaluation. Error: {ex}")
                    _finalNode = None

                    if isinstance(_initialValues, xr.DataArray):
                        _finalNode = self._tryFilter(
                            _resultNode, dynamicIndex, item) + self._tryFilter(_initialValues, dynamicIndex, item)
                    else:
                        _finalNode = self._tryFilter(
                            _resultNode, dynamicIndex, item) + _initialValues

                    cyclicDic[_id] = self.assignNewValues(
                        dynamicIndex.name, _finalNode, cyclicDic[_id], loc_dic)

                else:
                    try:
                        # dont use use initial values
                        _resultNode = evaluate(
                            _node["loopDefinition"], cyclicParams, False, fromDynamic=True)
                    except Exception as ex:
                        raise ValueError(
                            f"Node '{_id}' failed during dynamic evaluation. Error: {ex}")
                    _finalNode = self._tryFilter(
                        _resultNode, dynamicIndex, item)

                    cyclicDic[_id] = self.assignNewValues(
                        dynamicIndex.name, _finalNode, cyclicDic[_id], loc_dic)

                _node["calcTime"] += time.time() - evaluate_start_time

            # set dynamicVar
            if (not reverseMode and (nn+1) < initialCount) or (reverseMode and (nn-1) > initialCount):
                # do not update t- vars
                pass
            else:
                for _var in dynamicVars:
                    evaluate_start_time = time.time()

                    _key = "__" + _var + "_t"
                    if reverseMode:
                        cyclicDic[_key] = self._tryFilter(
                            cyclicDic[_var], dynamicIndex, dynamicIndex.values[nn+shift-1])
                    else:
                        cyclicDic[_key] = self._tryFilter(
                            cyclicDic[_var], dynamicIndex, dynamicIndex.values[nn-initialCount+1])

                    _node = cyclicNodes[nodesInCyclic.index(_var)]
                    _node["calcTime"] += time.time() - evaluate_start_time

        if sliceInputs:
            evaluate_start_time = time.time()
            for external_input_id, external_input in external_inputs.items():
                node.model.getNode(external_input_id)._result = external_input

            general_time += time.time() - evaluate_start_time

        general_time_proportion = general_time / len(nodesInCyclic)
        # set result
        for _node in cyclicNodes:
            _id = _node["node"].identifier
            _node["node"]._result = cyclicDic[_id]
            _node["node"]._isCalc = True
            _node["node"].lastEvaluationTime = _node["calcTime"] - \
                _node["node"].lastLazyTime + general_time_proportion
            _node["node"].evaluationVersion = node.model.evaluationVersion

        if node.model.debugMode:
            for circular_node_id in nodesInCyclic:
                circular_node = node.model.getNode(circular_node_id)
                if not circular_node is None:
                    node.model.sendEndCalcNode(circular_node_id, fromDynamic=node.identifier)

        evaluate = None
        model = None
        cyclicDic = None
        cyclicParams = None

        return "ok"

    def generateLoopDef(self, node, nodeDefinition, cyclicVariables):
        """Return definition used for circular evaluator"""
        _def = self.clearCircularDependency(
            nodeDefinition, "cyclicDic['__##node##_t']")
        finalDef = _def
        tmpCode = compile(_def, '<string>', 'exec')
        names = node.parseNames(tmpCode)
        rx = r"('[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")|\b{0}\b"
        for nodeId in names:
            if node._model.existNode(node._model.clearId(nodeId)):
                if nodeId in cyclicVariables:
                    finalDef = re.sub(rx.format(nodeId), lambda m:
                                      (
                        m.group(1)
                        if m.group(1)
                        else
                        (
                            "cyclicDic['"+nodeId+"']"
                            if (m.endpos > m.regs[0][1]+5) and ((m.string[m.regs[0][1]:m.regs[0][1]+5] == '.node') or (m.string[m.regs[0][1]:m.regs[0][1]+8] == '.timeit('))
                            else
                            (nodeId
                             if (m.string[m.regs[0][0]-1:m.regs[0][0]+len(nodeId)] == ('.'+nodeId))
                             else "self._tryFilter( cyclicDic['"+nodeId+"'],dynamicIndex,item) "
                             )
                        )
                    ), finalDef, 0, re.IGNORECASE)

        return finalDef

    def generateInitDef(self, node, nodeCube, dynamicIndex):
        """Return definition used for initialice vars in circular evaluator"""

        if isinstance(nodeCube, xr.DataArray):
            _list = list(nodeCube.dims[:])
            if not dynamicIndex.name in _list:
                _list.append(dynamicIndex.name)
            _dims = ','.join(_list)
            _def = f"result = pp.create_dataarray(0.,[{_dims}])"
            return _def
        return f"result = pp.create_dataarray(0.,[{dynamicIndex.name}])"

    def generateCircularParameters(self, node, nodeList):
        """Generate parameters for circularEval"""
        dynamicVars = []
        dynamicIndex = None
        nodesInCyclic = []  # nodeList TODO: Determinar orden de nodos
        initialValues = {}
        indexDic = {}
        shift = -1
        sliceInputs = False

        for _nodeId in nodeList:
            if node.model.existNode(_nodeId):
                node_def = node.model.getNode(_nodeId).definition
                params_list = self.getParametersFromDefinition(node_def)
                if params_list is not None:
                    # dataArray
                    dataArray_param = params_list[0]
                    if dataArray_param not in dynamicVars:
                        dynamicVars.append(dataArray_param)
                    # index
                    dynamicIndex = node.model.getNode(params_list[1]).result
                    if dynamicIndex.name not in indexDic:
                        indexDic[dynamicIndex.name] = _nodeId
                    # shift
                    shift = int(params_list[2])
                    # initialValues
                    if len(params_list) >= 4:
                        initialValues[_nodeId] = f'result = {params_list[3]}'
                    if len(params_list) >= 5:
                        # Set to True if there is at least one def with True
                        if not sliceInputs:
                            sliceInputs = params_list[4] == 'True'

        _graph = {}
        for _nodeId in nodeList:
            if node.model.existNode(_nodeId):
                _graph[_nodeId] = []
                for _outputId in node.model.getNode(_nodeId).ioEngine.outputs:
                    if _outputId in nodeList:
                        _output_def = str(node.model.getNode(
                            _outputId).definition).replace(' ', '')
                        if f'dynamic({_nodeId},' not in _output_def:
                            _graph[_nodeId].append(_outputId)

        nodesInCyclic = Helpers.dfs_topsort(_graph)

        return {
            'dynamicVars': dynamicVars,
            'dynamicIndex': dynamicIndex,
            'nodesInCyclic': nodesInCyclic,
            'initialValues': initialValues,
            'indexDic': indexDic,
            'shift': shift,
            'sliceInputs': sliceInputs

        }

    def clearCircularDependency(self, definition: str, replace_with: str = '0') -> str:
        """Replaces pp.dynamic(x,y,z) with the desired replace_with param"""
        new_def = definition
        replacement_value = replace_with

        dynamic_positions = self.getPositionsFromDefinition(definition)
        if dynamic_positions is not None:
            dynamic_initial_pos, dynamic_last_pos = dynamic_positions
            dynamic_def = definition[dynamic_initial_pos:dynamic_last_pos+1]

            if 'cyclicDic' in replace_with:
                # Get dataArray parameter
                dynamic_params = self.getParametersFromDefinition(
                    definition)
                if dynamic_params is not None:
                    # Replace dataArray parameter
                    dataArray_param = dynamic_params[0]
                    replacement_value = replacement_value.replace(
                        '##node##', dataArray_param)

            new_def = new_def.replace(dynamic_def, replacement_value)

        return new_def

    def _tryFilter(self, array, dim, value):
        try:
            _dic = {
                dim.name: value
            }
            return array.sel(_dic, drop=True)
        except Exception as ex:
            return array

    def assignNewValues(
        self,
        dynamic_index_name: str,
        new_values: xr.DataArray,
        destination: xr.DataArray,
        loc_dic: dict
    ):
        destination_array = destination
        new_values_array = new_values

        destination_array = self.reshapeDataArrays(
            dynamic_index_name, destination_array, list(new_values_array.dims))

        try:
            destination_array.loc[loc_dic] = new_values_array.values
        except Exception as ex:
            try:
                # Add new dims in destination array
                new_dims_dict = {
                    dim: new_values_array.coords[dim] for dim in new_values_array.dims if dim not in destination_array.dims}
                if new_dims_dict:
                    destination_array = destination_array.expand_dims(
                        new_dims_dict).copy()  # copying to avoid view error
                # Get all dims in both arrays and reorder them with dynamic_index_name first
                all_dims = set(destination_array.dims)
                all_dims.update(set(new_values_array.dims))
                all_dims_list = list(all_dims)
                all_dims_list.remove(dynamic_index_name)
                all_dims_reordered = [
                    dynamic_index_name] + all_dims_list
                # Reshape destination and new_values arrays so that both are in the same order
                destination_array = destination_array.transpose(
                    *[dim for dim in all_dims_reordered if dim in destination_array.dims], transpose_coords=True)
                new_values_array = new_values_array.transpose(
                    *[dim for dim in all_dims_reordered if dim in new_values_array.dims], transpose_coords=True)
                # Assign new values to destination array
                destination_array.loc[loc_dic] = new_values_array.values
            except Exception as ex:
                list_dims = list(destination_array.dims)
                list_dims.remove(dynamic_index_name)
                destination_array.loc[loc_dic] = new_values_array.transpose(
                    *list_dims, transpose_coords=True).values

        return destination_array

    def reshapeDataArrays(
        self,
        dynamic_index_name: str,
        destination: xr.DataArray,
        new_values_dims: list
    ) -> xr.DataArray:
        """Reshape dimensions and coords to avoid problems when
           there are 2 or more dims with same number of elements
        """
        destination_array = destination

        destination_dims = list(destination_array.dims)
        destination_dims.remove(dynamic_index_name)
        if len(destination_dims) == len(new_values_dims) and destination_dims != new_values_dims:
            # This should happen only after first assignment; afterwards, it will have the correct shape
            reordered_dims = [dynamic_index_name] + new_values_dims
            destination_array = destination_array.transpose(
                *reordered_dims).copy()  # copying to avoid view error

        return destination_array
