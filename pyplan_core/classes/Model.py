import datetime
import importlib
import os
import re
import subprocess
import sys
import threading
import unicodedata
from copy import deepcopy
from shlex import split
from site import getsitepackages
from sys import platform
from time import sleep

import jsonpickle
import numpy
import pandas
import xarray as xr

from pyplan_core import cubepy
from pyplan_core.classes.BaseNode import BaseNode
from pyplan_core.classes.evaluators.Evaluator import Evaluator
from pyplan_core.classes.Intellisense import Intellisense
from pyplan_core.classes.IOModule import IOModule
from pyplan_core.classes.PyplanFunctions import PyplanFunctions, Selector
from pyplan_core.classes.wizards import (CalculatedField, DataarrayFromPandas,
                                         DataframeGroupby, DataframeIndex,
                                         InputTable, SelectColumns, SelectRows,
                                         sourcecsv, CreateIndex, IndexFromPandas, DataArrayFilter, EditIndex, RenameIndexItem)
from pyplan_core.classes.ws.settings import ws_settings

from .DefaultNodeFormats import default_formats


class Model(object):

    DEFAULT_IMPORTS = {
        'np': numpy,
        'pd': pandas,
        'cubepy': cubepy,
        'xr': xr
    }

    def __init__(self, WSClass=None):
        self._nodeDic = {}
        self._modelProp = {}
        self._modelNode = None
        self._isLoadingModel = False
        self.initialize()
        self.evaluationVersion = 0
        self.inCyclicEvaluate = False
        self._scenarioDic = dict()
        self._wizard = None
        self._currentProcessingNode = ''
        self._currentInstallProgress = []
        self._customImports = None
        self.company_code = None
        self.session_key = None
        self.debugMode = None
        self.WS = WSClass

    # Props

    @property
    def nodeDic(self):
        return self._nodeDic

    @property
    def modelProp(self):
        return self._modelProp

    @property
    def modelNode(self):
        return self._modelNode

    @property
    def isLoadingModel(self):
        return self._isLoadingModel

    # Methods

    def getPID(self):
        return os.getpid()

    def getDefaultNodeFormat(self, nodeClass):
        default_formats
        if nodeClass in default_formats:
            return deepcopy(default_formats[nodeClass])
        return None

    def getCurrentModelPath(self):
        if self.existNode('current_path'):
            return self.getNode('current_path').result
        return ''

    def setCurrentModelPath(self, value):
        if self.existNode('current_path'):
            self.getNode('current_path').definition = 'result="""' + \
                str(value) + '"""'

    def currentProcessingNode(self, nodeId):
        if nodeId not in ['__evalnode__', 'current_path']:
            self._currentProcessingNode = nodeId

    def initialize(self, modelName=None):

        if modelName is None:
            self._modelNode = self.createNode('new_model', 'model', '_model_')
        else:
            newId = modelName.lower()
            newId = re.sub('[^0-9a-z]+', '_', newId)
            self._modelNode = self.createNode(newId, 'model', '_model_')
            self._modelNode.title = modelName

        self._scenarioDic = dict()
        self._wizard = None

    def connectToWS(self, company_code, session_key):
        # Connect to WS
        self.company_code = company_code
        self.session_key = session_key
        if self.WS:
            self.ws = self.WS(company_code=company_code,
                              session_key=session_key)

    def createNode(self, identifier=None, nodeClass=None, moduleId=None, x=None, y=None, toObj=False, originalId=None):
        """Create new node"""
        newNode = BaseNode(model=self, identifier=identifier, nodeClass=nodeClass,
                           moduleId=moduleId, x=x, y=y, originalId=originalId)
        id = newNode.identifier
        self.nodeDic[id] = newNode
        newNode = None
        if toObj:
            return self.nodeDic[id].toObj()
        else:  # uso interno
            return self.nodeDic[id]

    def deleteNodes(self, nodes, removeAliasIfNotIn=None):
        """Delete nodes by node id"""
        if not nodes is None:
            for nodeId in nodes:
                if self.existNode(nodeId) and nodeId != '_model_':
                    # check for module
                    if self.getNode(nodeId).nodeClass == 'module':
                        childs = self.findNodes('moduleId', nodeId)
                        childsIds = [c.identifier for c in childs]
                        self.deleteNodes(childsIds, removeAliasIfNotIn)

                    # check for aliases
                    aliases = []
                    aliases = self.findNodes('originalId', nodeId)
                    if aliases:
                        aliasesId = [a.identifier for a in aliases]
                        # check for aliases in other modules
                        if removeAliasIfNotIn:
                            _auxAliases = []
                            for _aliasId in aliasesId:
                                if self.isIn(_aliasId, removeAliasIfNotIn):
                                    _auxAliases.append(_aliasId)
                            if len(_auxAliases) > 0:
                                self.deleteNodes(
                                    _auxAliases, removeAliasIfNotIn)
                        else:
                            self.deleteNodes(aliasesId, removeAliasIfNotIn)

                    self.nodeDic[nodeId].preDelete()
                    self.nodeDic[nodeId].release()
                    self.nodeDic[nodeId] = None
                    del self.nodeDic[nodeId]
        pass

    def existNode(self, nodeId):
        """Return True if node exists in model"""
        nodeId = self.clearId(nodeId)
        if (not nodeId is None) & (nodeId in self.nodeDic):
            return True
        return False

    def getNode(self, nodeId):
        """Renor node from node dictionary"""
        if self.existNode(nodeId):
            return self.nodeDic[nodeId]

    def isChild(self, nodeId, modulesId):
        """Return true if nodeid or one of your parents is in any of modulesId modules"""
        res = False
        if self.existNode(nodeId):
            aux = self.getNode(nodeId).moduleId
            nChance = 20
            while (res == False and aux != '_model_' and nChance > 0):
                res = aux in modulesId
                node = self.getNode(aux)
                if node:
                    aux = node.moduleId
                else:
                    nChance = 0
                nChance -= 1
        return res

    def getAPlace(self, moduleId, x, y, w, h, deltaX=110):
        """ Get a place for insert a node """
        res = dict(x=x, y=y)
        options = 100
        l1x = int(x)
        l1y = int(y)
        r1x = int(x) + int(w)
        r1y = int(y) + int(h)
        endy = int(y) + int(h)
        while options > 0:
            options -= 1
            found = False
            for node in self.findNodes('moduleId', moduleId):
                if node.nodeClass != "text":
                    l2x = int(node.x)
                    l2y = int(node.y)
                    r2x = int(node.x) + int(node.w)
                    r2y = int(node.y) + int(node.h)
                    if not (l1x > r2x or l2x > r1x or l1y > r2y or l2y > r1y):
                        found = True
                        break
            if found:
                l1x = l1x + deltaX
                r1x = r1x + deltaX
            else:
                res["x"] = l1x
                res["y"] = l1y
                break
        return res

    def isNodeInScenario(self, nodeId):
        return nodeId in self._scenarioDic

    def evaluateNode(self, nodeId, dims=None, rows=None, columns=None, summaryBy='sum',
                     bottomTotal=False, rightTotal=False, fromRow=0, toRow=0, hideEmpty=None,
                     rowOrder='original', columnOrder='original'):
        """Evaluate node. Call evaluator class for implement diferent evaluators."""
        if self.existNode(nodeId):
            result = None
            if nodeId in self._scenarioDic:
                result = self._scenarioDic[nodeId]
            else:
                result = self.nodeDic[nodeId].result

            if not result is None:
                self.evaluationVersion += 1
                evaluator = Evaluator.createInstance(result)
                return evaluator.evaluateNode(result, self.nodeDic, nodeId, dims, rows, columns,
                                              summaryBy, bottomTotal, rightTotal, fromRow, toRow,
                                              hideEmpty, rowOrder, columnOrder)
        return ''

    def executeButton(self, nodeId):
        """Execute node of class button"""
        if self.existNode(nodeId):
            self.nodeDic[nodeId].invalidate()
            toReturn = self.nodeDic[nodeId].result
            if toReturn is None:
                toReturn = ''
            return toReturn

        return ''

    def previewNode(self, nodeId, debugMode=""):
        """Perform preview of a node"""
        try:
            result = None
            if self.existNode(nodeId):
                if debugMode:
                    self.debugMode = debugMode
                    if debugMode == "node":
                        self.nodeDic[nodeId].silentInvalidate()
                    elif debugMode == "model":
                        for node_key in self.nodeDic:
                            self.nodeDic[node_key].silentInvalidate()

                if not self.nodeDic[nodeId].originalId is None:
                    nodeId = self.nodeDic[nodeId].originalId
                if self.nodeDic[nodeId].nodeClass in ["button", "module", "text"]:
                    self.evaluationVersion += 1
                    _dummy = self.nodeDic[nodeId].result  # for execute button
                elif not self.nodeDic[nodeId].result is None:
                    self.evaluationVersion += 1
                    evaluator = Evaluator.createInstance(
                        self.nodeDic[nodeId].result)
                    result = evaluator.previewNode(self.nodeDic, nodeId)
            if result is None:
                evaluator = Evaluator.createInstance(None)
                result = evaluator.generateEmptyPreviewResponse(
                    self.nodeDic, nodeId)
            return result
        finally:
            if self.debugMode and self.ws:
                self.ws.sendDebugInfo("endPreview", "", "endPreview")
            self.debugMode = None

    def getCubeValues(self, query):
        """Evaluate node. Used for pivotgrid"""
        nodeId = query['cube']
        if self.existNode(nodeId):
            result = None
            if nodeId in self._scenarioDic:
                result = self._scenarioDic[nodeId]
            else:
                result = self.nodeDic[nodeId].result

            if not result is None:
                evaluator = Evaluator.createInstance(result)
                return evaluator.getCubeValues(result, self.nodeDic, nodeId, query)

    def getCubeDimensionValues(self, query):
        """Return the values of a dimension of node. Used from pivotgrid"""
        nodeId = query['cube']
        if self.existNode(nodeId):
            result = None
            if nodeId in self._scenarioDic:
                result = self._scenarioDic[nodeId]
            else:
                result = self.nodeDic[nodeId].result

            if not result is None:
                evaluator = Evaluator.createInstance(result)
                return evaluator.getCubeDimensionValues(result, self.nodeDic, nodeId, query)

    def getCubeMetadata(self, nodeId):
        """Return metadata of cube. Used from pivotgrid"""
        if self.existNode(nodeId):
            result = None
            if nodeId in self._scenarioDic:
                result = self._scenarioDic[nodeId]
            else:
                result = self.nodeDic[nodeId].result

            if not result is None:
                evaluator = Evaluator.createInstance(result)
                return evaluator.getCubeMetadata(result, self.nodeDic, nodeId)

    def setNodeValueChanges(self, changes):
        """Set values for node using filters"""
        nodeId = changes['node']
        if self.existNode(nodeId):
            if self.nodeDic[nodeId].nodeClass == 'formnode':
                nodeId = self.nodeDic[nodeId].originalId
                evaluator = Evaluator.createInstance(None)
                return evaluator.setNodeValueChanges(self.nodeDic, nodeId, changes)
            else:
                if not self.nodeDic[nodeId].originalId is None:
                    nodeId = self.nodeDic[nodeId].originalId
                if not self.nodeDic[nodeId].result is None:
                    evaluator = Evaluator.createInstance(
                        self.nodeDic[nodeId].result)
                    return evaluator.setNodeValueChanges(self.nodeDic, nodeId, changes)

    def getDiagram(self, moduleId=None):
        """Get diagram"""
        if moduleId is None:
            moduleId = self.modelNode.identifier
        moduleId = self.clearId(moduleId)
        res = {
            'moduleId': moduleId,
            'arrows': [],
            'nodes': [],
            'breadcrumb': self.getBreadcrumb(moduleId)
        }
        nodeList = self.findNodes('moduleId', moduleId)
        nodeList.sort(key=lambda x: int(x.z))
        for node in nodeList:
            exceptions = ['definition'] if node.nodeClass == 'text' else [
                'definition', 'description']
            res['nodes'].append(node.toObj(
                exceptions=exceptions, fillDefaultProperties=True))
        return res

    def getBreadcrumb(self, moduleId=None):
        """Get breadcrumb"""
        if moduleId is None:
            moduleId = self.modelNode.identifier
        moduleId = self.clearId(moduleId)
        res = []
        aux = moduleId
        while aux != self.modelNode.identifier and self.existNode(aux):
            res.append({
                'identifier': aux,
                'title': (self.getNode(aux).title or aux)
            })
            aux = self.getNode(aux).moduleId

        res.append({'identifier': self.modelNode.identifier,
                    'title': self.modelNode.title or 'Main'})
        return res

    def isIn(self, nodeId, moduleId):
        """ Return true if nodeId is in moduleId. Search for parents"""
        res = False
        aux = nodeId
        _secure = 1
        while aux != self.modelNode.identifier and self.existNode(aux) and _secure < 100:
            if aux == moduleId:
                res = True
                break
            aux = self.getNode(aux).moduleId
            _secure += 1
        return res

    def isSelector(self, nodeId):
        """Return True if nodeId is of type Selector"""
        if self.existNode(nodeId):
            return isinstance(self.getNode(nodeId).result, Selector)
        return False

    def getSelector(self, nodeId):
        """Return selector data if node is of type selector"""
        if self.isSelector(nodeId):
            return self.getNode(nodeId).result.toObj()

    def setSelectorValue(self, nodeId, value):
        """Set selector value if node is of type selector"""
        if self.isSelector(nodeId):

            node = self.getNode(nodeId)
            selector = node.result

            if not selector.isSameValue(value):
                definition = node.definition
                new_definition = selector.generateDefinition(definition, value)
                if new_definition:
                    node.definition = new_definition

    def getSelectorValue(self, nodeId):
        """Return selector value if node is of type selector"""
        if self.isSelector(nodeId):
            return self.getNode(nodeId).result.value

    def release(self):
        """Release model. Free all resources """
        if not self._modelNode is None:
            self._modelNode.release()

        (xx.release() for xx in self.nodeDic)
        keys = [x for x in self.nodeDic]
        for key in keys:
            del self.nodeDic[key]

        self._nodeDic = {}
        self._modelProp = {}
        self._modelNode = None
        self._scenarioDic = dict()
        self._wizard = None
        self._customImports = None

        return

    def getNextIdentifier(self, prefix):
        """Get next free identifier of node"""
        reg = r'(\d+$)'
        matches = re.findall(reg, prefix)
        start_at = 1
        if len(matches) > 0:
            num = matches[0]
            start_at = int(num)+1
            if start_at > 100000000:
                prefix += '_'
                start_at = 1
            else:
                prefix = prefix[: -len(num)]

        for num in range(start_at, 100000000):
            key = prefix+str(num)
            if not key in self.nodeDic:
                return key

    def clearId(self, nodeId):
        """DEPRECATED"""
        # if not nodeId is None:
        #     return nodeId.lower()
        return nodeId

    def updateNodeIdInDic(self, oldNodeId, newNodeId):
        """Update node identifier on all dictionary."""
        if self.existNode(oldNodeId):
            self.nodeDic[newNodeId] = self.nodeDic[oldNodeId]
            for node in self.findNodes('moduleId', oldNodeId):
                node.moduleId = newNodeId
            for node in self.findNodes('originalId', oldNodeId):
                node.originalId = newNodeId
                node._definition = 'result = ' + str(newNodeId)
            del self.nodeDic[oldNodeId]
            return True
        return False

    def setNodeProperties(self, nodeId, properties):
        """Update properties of a node"""
        nodeId = self.clearId(nodeId)
        if self.existNode(nodeId):
            _node = self.getNode(nodeId)
            for prop in properties:
                if '.' in prop['name']:
                    nodeProp, objProp = prop['name'].split('.')
                    setattr(getattr(_node, nodeProp), objProp, prop['value'])
                else:
                    setattr(_node, prop['name'], prop['value'])

    def getNodeProperties(self, nodeProperties):
        """Get properties of a node"""
        if (not nodeProperties is None) and (nodeProperties['node'] != ''):
            nodeId = self.clearId(nodeProperties['node'])
            if self.existNode(nodeId):
                _node = self.getNode(nodeId)
                for prop in nodeProperties['properties']:
                    if hasattr(_node, prop['name']):
                        prop['value'] = getattr(_node, prop['name'])
                return nodeProperties
        pass

    def setModelProperties(self, properties):
        """Update properties of model"""
        for key in properties:
            if key == 'modelId':
                continue
            elif key in ['identifier', 'title']:
                setattr(self.modelNode, key, properties[key])
            else:
                self.modelProp[key] = properties[key]

    def getModelProperties(self):
        """Get model propierties"""
        res = dict()
        # fill model id and tile
        res['identifier'] = self.modelNode.identifier
        res['title'] = self.modelNode.title

        for key in self.modelProp:
            res[key] = self.modelProp[key]

        return res

    def getIndexes(self, nodeId):
        """Return indexes of a node"""
        if self.existNode(nodeId):
            if nodeId in self._scenarioDic:
                evaluator = Evaluator.createInstance(self._scenarioDic[nodeId])
                return evaluator.getIndexes(self.nodeDic[nodeId], self._scenarioDic[nodeId])
            else:
                _node = self.getNode(nodeId)
                return _node.indexes

    def getIndexValues(self, data):
        """Return values of a index node."""
        tmpNodeId = data.index_id if data.node_id is None or data.node_id == '' else data.node_id
        if self.existNode(tmpNodeId):
            result = self.nodeDic[tmpNodeId].result
            if data.node_id in self._scenarioDic:
                result = self._scenarioDic[data.node_id]
            evaluator = Evaluator.createInstance(result)
            return evaluator.getIndexValues(self.nodeDic, data, result)

    def getIndexType(self, nodeId, indexId):
        """Return index type"""
        tmpNodeId = indexId if nodeId is None or nodeId == '' else nodeId
        if self.existNode(tmpNodeId):
            evaluator = Evaluator.createInstance(
                self.nodeDic[tmpNodeId].result)
            return evaluator.getIndexType(self.nodeDic, nodeId, indexId)

    def getIndexesWithLevels(self, nodeId):
        """Return indexes of a node"""
        if self.existNode(nodeId):
            if nodeId in self._scenarioDic:
                evaluator = Evaluator.createInstance(self._scenarioDic[nodeId])
                return evaluator.getIndexesWithLevels(self.nodeDic[nodeId], self._scenarioDic[nodeId])
            else:
                evaluator = Evaluator.createInstance(
                    self.nodeDic[nodeId].result)
                return evaluator.getIndexesWithLevels(self.nodeDic[nodeId])

    def isTable(self, nodeId):
        """return true if node is a table"""
        res = '0'
        if self.existNode(nodeId):
            evaluator = Evaluator.createInstance(self.nodeDic[nodeId].result)
            res = evaluator.isTable(self.getNode(nodeId))
        return res

    def getArrows(self, moduleId):
        """Return all arrows of moduleId"""
        res = []
        modulesInLevel = []
        inputsInOtherLevel = []
        outputsInOtherLevel = []
        thisLevel = self.findNodes('moduleId', moduleId)
        thisIds = [node.identifier for node in thisLevel]
        for node in thisLevel:
            if node.nodeClass == 'module':
                modulesInLevel.append(node.identifier)

        # node to node
        for node in thisLevel:
            if node.nodeClass not in ['module', 'text']:
                for outputNodeId in node.outputs:
                    # aliases
                    fullOutputs = []
                    fullOutputs = self.findNodes('originalId', outputNodeId)
                    fullOutputs.append(self.getNode(outputNodeId))
                    for o in fullOutputs:
                        if not o is None:
                            element = {'from': node.identifier,
                                       'to': o.identifier}
                            if o.identifier in thisIds:
                                if node.nodeInfo.showOutputs and o.nodeInfo.showInputs:
                                    if self.existArrow(element['from'], element['to'], res) == False:
                                        res.append(element)
                            elif o.identifier not in thisIds:
                                if self.existArrow(element['from'], element['to'], outputsInOtherLevel) == False:
                                    # if theres an alias in this level don't include the arrow
                                    if not len(self.getAliasInLevel(o.identifier, moduleId)) > 0:
                                        outputsInOtherLevel.append(element)
                for inputNodeId in node.inputs:
                    # aliases
                    fullInputs = []
                    fullInputs = self.findNodes('originalId', inputNodeId)
                    fullInputs.append(self.getNode(inputNodeId))
                    for i in fullInputs:
                        if not i is None:
                            element = {'from': i.identifier,
                                       'to': node.identifier}
                            if i.identifier in thisIds:
                                if i.nodeInfo.showOutputs and node.nodeInfo.showInputs:
                                    if self.existArrow(element['from'], element['to'], res) == False:
                                        res.append(element)
                            elif i.identifier not in thisIds:
                                if self.existArrow(element['from'], element['to'], inputsInOtherLevel) == False:
                                    # if theres an alias in this level don't include the arrow
                                    if not len(self.getAliasInLevel(i.identifier, moduleId)) > 0:
                                        inputsInOtherLevel.append(element)

        # node to module
        if outputsInOtherLevel:
            for d in outputsInOtherLevel:
                newTo = []
                nodeFrom = d['from']
                nodeTo = d['to']
                if self.getNode(nodeTo).isin in self.nodeDic:
                    newTo = self.getParentModule(nodeTo, modulesInLevel)
                if newTo:
                    element = {'from': nodeFrom, 'to': newTo}
                    if self.getNode(nodeFrom).nodeInfo.showOutputs and self.getNode(newTo).nodeInfo.showInputs:
                        if self.existArrow(element['from'], element['to'], res) == False:
                            res.append(element)

        # module to node
        if inputsInOtherLevel:
            for d in inputsInOtherLevel:
                newFrom = []
                nodeFrom = d['from']
                nodeTo = d['to']
                if self.getNode(nodeFrom).isin in self.nodeDic:
                    newFrom = self.getParentModule(nodeFrom, modulesInLevel)
                if newFrom:
                    element = {'from': newFrom, 'to': nodeTo}
                    if self.getNode(newFrom).nodeInfo.showOutputs and self.getNode(nodeTo).nodeInfo.showInputs:
                        if self.existArrow(element['from'], element['to'], res) == False:
                            res.append(element)

        # module to module
        modulesComplete = []
        for mod in modulesInLevel:
            modulesComplete.append(
                {'module': mod, 'nodes': self.getNodesInModule(mod, [])})

        for mod in modulesComplete:
            for node in mod['nodes']:
                if self.getNode(mod['module']).nodeInfo.showOutputs:
                    tempOutputs = node.outputs
                    if tempOutputs:
                        for output in tempOutputs:
                            for auxModule in modulesComplete:
                                if(mod['module'] != auxModule['module'] and self.getNode(auxModule['module']).nodeInfo.showInputs):
                                    # module to module
                                    if self.getNode(output) in auxModule['nodes']:
                                        element = {
                                            'from': mod['module'], 'to': auxModule['module']}
                                        if self.existArrow(element['from'], element['to'], res) == False:
                                            res.append(element)
                                """# module to alias
                                aliases = self.getAliasInLevel(
                                    output, moduleId)
                                if len(aliases) > 0:
                                    for alias in aliases:
                                        if alias.nodeInfo.showInputs:
                                            element = {
                                                "from": mod["module"], "to": alias.identifier}
                                            if self.existArrow(element["from"], element["to"], res) == False:
                                                res.append(element)
                                            # alias to alias
                                            outputAliases = self.getAliasInLevel(
                                                node.identifier, moduleId)
                                            if len(outputAliases) > 0:
                                                for outAl in outputAliases:
                                                    if outAl.nodeInfo.showOutputs:
                                                        element = {
                                                            "from": outAl.identifier, "to": alias.identifier}
                                                        if self.existArrow(element["from"], element["to"], res) == False:
                                                            res.append(element)

                # alias to module
                if self.getNode(mod["module"]).nodeInfo.showInputs:
                    tempInputs = node.inputs
                    if tempInputs:
                        for inp in tempInputs:
                            aliases = self.getAliasInLevel(inp, moduleId)
                            if len(aliases) > 0:
                                for auxModule in modulesComplete:
                                    for alias in aliases:
                                        if alias.nodeInfo.showOutputs:
                                            element = {
                                                "from": alias.identifier, "to": mod["module"]}
                                            if self.existArrow(element["from"], element["to"], res) == False:
                                                res.append(element)"""

        return res

    def existArrow(self, aFrom, aTo, arrowsList):
        """Return true if exists de arrow from-to"""
        if arrowsList:
            for arrow in arrowsList:
                if arrow['from'] == aFrom and arrow['to'] == aTo:
                    return True
        return False

    def getAliasInLevel(self, nodeIdentifier, levelId):
        """Returns the aliases in the level"""
        res = []
        aliasNodes = self.findNodes('originalId', nodeIdentifier)
        if aliasNodes is not None:
            for alias in aliasNodes:
                if alias.moduleId == levelId:
                    res.append(alias)
                    break
        return res

    def getParentModulesWithAlias(self, moduleId, modulesArray):
        """Return the parent module with alias"""
        if moduleId != '_model_':
            if moduleId not in modulesArray:
                modulesArray.append(moduleId)
            alias = []
            alias = self.findNodes('originalId', moduleId)
            if alias:
                for a in alias:
                    if a.identifier not in modulesArray:
                        modulesArray.append(a.identifier)
            if self.getNode(moduleId).isin in self.nodeDic:
                return self.getParentModulesWithAlias(self.getNode(moduleId).isin, modulesArray)
            else:
                return modulesArray
        else:
            return modulesArray

    def getParentModule(self, moduleId, modulesInLevel):
        """Return parent module"""
        if moduleId == '_model_':
            return None
        else:
            if moduleId in modulesInLevel:
                return moduleId
            else:
                if self.getNode(moduleId).isin in self.nodeDic:
                    return self.getParentModule(self.getNode(moduleId).isin, modulesInLevel)
                else:
                    return None

    def getNodesInModule(self, moduleId, nodesInSubLevels):
        """Return nodes un module"""

        subLevelNodes = []
        modulesInSubLevels = []
        subLevelNodes = self.findNodes('moduleId', moduleId)
        if subLevelNodes:
            for node in subLevelNodes:
                if node.nodeClass == 'module':
                    modulesInSubLevels.append(node)
                else:
                    if node.nodeClass != 'text':
                        nodesInSubLevels.append(node)

            if modulesInSubLevels:
                for module in modulesInSubLevels:
                    self.getNodesInModule(module.identifier, nodesInSubLevels)

        return nodesInSubLevels

    # UNUSED FULL ARROWS (NODE TO NODE WITH ALIASES)
    """
    def getArrows(self,moduleId):
        res=[]
        thisLevel = self.findNodes("moduleId",moduleId)
        thisIds = [node.identifier for node in thisLevel]
        for nodeId in self.nodeDic:
            if self.getNode(nodeId).nodeClass not in ["module","text"]:
                nodeInputs = []
                nodeOutputs = []
                fullNode = []
                fullInputs = []
                fullOutputs = []
                # To clear non existent aliases
                if self.getNode(nodeId).originalId:
                    if self.getNode(nodeId).originalId not in self.nodeDic:
                        continue
                nodeInputs = self.getNode(nodeId).inputs
                nodeOutputs = self.getNode(nodeId).outputs
                fullNode = self.getNodeParentsAndAliases(nodeId)
                if fullNode:
                    for n in fullNode:
                        if n not in thisIds:
                            fullNode.remove(n)
                if fullNode:
                    if nodeInputs:
                        for i in nodeInputs:
                            fullInputs = self.getNodeParentsAndAliases(i)
                            if fullInputs:
                                for n in fullInputs:
                                    if n not in thisIds:
                                        fullInputs.remove(n)
                            if fullInputs:
                                for n in fullNode:
                                    for inp in fullInputs:
                                        if self.getNode(n).nodeInfo.showInputs and self.getNode(inp).nodeInfo.showOutputs:
                                            element = {"from":inp,"to":n}
                                            if element not in res:
                                                res.append(element)

                    if nodeOutputs:
                        for o in nodeOutputs:
                            fullOutputs = self.getNodeParentsAndAliases(o)
                            if fullOutputs:
                                for n in fullOutputs:
                                    if n not in fullOutputs:
                                        fullOutputs.remove(n)
                            if fullOutputs:
                                for n in fullNode:
                                    for out in fullOutputs:
                                        if self.getNode(out).nodeInfo.showInputs and self.getNode(n).nodeInfo.showOutputs:
                                            element = {"from":n,"to":out}
                                            if element not in res:
                                                res.append(element)

        return res


    def getNodeParentsAndAliases(self,nodeId):
        response = []
        aliases = []
        parentModulesAndAliases = []
        if nodeId not in response:
            response.append(nodeId)
        aliases = self.findNodes('originalId',nodeId)
        if aliases:
            for a in aliases:
                if a.identifier not in response:
                    response.append(a.identifier)
        if self.getNode(nodeId).isin in self.nodeDic:
            parentModulesAndAliases = self.getParentModulesWithAlias(
                self.getNode(nodeId).isin,[])
        if parentModulesAndAliases:
            for m in parentModulesAndAliases:
                if m not in response:
                    response.append(m)
        return response

    """

    def findNodes(self, prop, value):
        """Finds nodes by property/value"""
        res = []
        for k, v in self.nodeDic.items():
            if getattr(v, prop) == value:
                if(not v.system):
                    res.append(self.nodeDic[k])
        return res

    def searchNodes(self, filterOptions):
        """Search nodes using filter options """
        res = []
        res = Intellisense().search(self, filterOptions)
        return res

    def getInputs(self, nodeId):
        res = []
        if self.existNode(nodeId):
            for nodeInput in self.getNode(nodeId).inputs:
                if self.existNode(nodeInput):
                    inp = self.getNode(nodeInput)
                    res.append({
                        'id': nodeInput,
                        'name': inp.title if not inp.title is None else nodeInput,
                        'nodeClass': inp.nodeClass,
                        'moduleId': inp.moduleId
                    })

        return res

    def getOutputs(self, nodeId):
        """Get output list of a node"""
        res = []
        if self.existNode(nodeId):
            for nodeOutput in self.getNode(nodeId).outputs:
                if self.existNode(nodeOutput):
                    out = self.getNode(nodeOutput)
                    res.append({
                        'id': nodeOutput,
                        'name': out.title if not out.title is None else nodeOutput,
                        'nodeClass': out.nodeClass,
                        'moduleId': out.moduleId
                    })
        return res

    def moveNodes(self, nodeList, moduleId):
        """Move nodes to other moduleId"""
        res = []
        moduleId = self.clearId(moduleId)
        if self.existNode(moduleId):
            for nodeId in nodeList:
                if self.existNode(nodeId) and nodeId != moduleId:
                    self.getNode(nodeId).moduleId = moduleId
                    res.append(nodeId)
        return res

    def copyNodes(self, nodeList, moduleId):
        """Copy nodes"""
        res = []
        if self.existNode(moduleId):
            try:
                self._isLoadingModel = True
                rx = r"('[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")|\b{0}\b"
                newNodesDic = dict()

                def nodeCreator(_nodeList, _moduleId):

                    for nodeId in _nodeList:
                        nodeId = self.clearId(nodeId)
                        if self.existNode(nodeId):
                            obj = self.getNode(nodeId).toObj()

                            newId = self.getNextIdentifier(
                                f"{obj['identifier']}")
                            newNodesDic[obj['identifier']] = newId
                            obj['identifier'] = newId

                            if obj['moduleId'] == _moduleId:
                                obj['x'] = int(obj['x']) + 10
                                obj['y'] = int(obj['y']) + 10
                            else:
                                obj['moduleId'] = _moduleId
                            nodeObj = self.createNode(
                                obj['identifier'], moduleId=_moduleId)
                            nodeObj.fromObj(obj)
                            res.append(nodeObj.identifier)

                            if nodeObj.nodeClass == 'module':
                                _childrens = [
                                    node.identifier for node in self.findNodes('moduleId', nodeId)]
                                nodeCreator(_childrens, newId)

                nodeCreator(nodeList, moduleId)

                # update definitions and alias original
                for sourceNode, targetNode in newNodesDic.items():
                    newNode = self.getNode(targetNode)
                    currentDef = newNode.definition
                    if not currentDef is None and currentDef != '':
                        tmpCode = newNode.compileDef(currentDef)
                        if not tmpCode is None:
                            names = newNode.parseNames(tmpCode)
                            for node in names:
                                if node in newNodesDic:
                                    newRelatedId = newNodesDic[node]

                                    currentDef = re.sub(rx.format(node), lambda m:
                                                        (
                                        m.group(1)
                                        if m.group(1)
                                        else
                                        newRelatedId
                                    ), currentDef, 0, re.IGNORECASE)
                                    # end parse definition
                            newNode.definition = currentDef

                    # update allias
                    if not newNode.originalId is None and newNode.originalId in newNodesDic:
                        newNode.originalId = newNodesDic[newNode.originalId]

                    # regenerate
                    newNode.generateIO()

            finally:
                self._isLoadingModel = False

        return res

    def copyAsValues(self, nodeId, asNewNode=False, aliasNode=None):
        """ Copy node as values """
        if self.existNode(nodeId):
            node = self.nodeDic[nodeId]
            if node.originalId:
                return self.copyAsValues(node.originalId, asNewNode, node)

            result = node.result
            if asNewNode:
                moduleId = aliasNode.moduleId if aliasNode is not None else node.moduleId
                x = aliasNode.x if aliasNode is not None else node.x
                y = aliasNode.y if aliasNode is not None else node.y
                newNode = self.createNode(identifier=self.getNextIdentifier(f'{nodeId}_copy'),
                                          moduleId=moduleId, nodeClass=node.nodeClass, x=int(x)+40, y=int(y)+60)
                newNode.w = node.w
                newNode.h = node.h
                newNode.definition = node.definition
                newNode.title = f'{node.title} copy' if node.title else node.identtifier
                node = newNode
            evaluator = Evaluator.createInstance(result)
            return evaluator.copyAsValues(result, self.nodeDic, node.identifier)

        return False

    def createInputNodes(self, nodeList):
        """Create input nodes"""
        res = []
        if not nodeList is None:
            for nodeId in nodeList:
                if self.existNode(nodeId):
                    firstNode = self.getNode(nodeId)
                    nodeOrig = self.getOriginalNode(nodeId)
                    inputNode = self.createNode(moduleId=firstNode.moduleId, nodeClass='formnode', x=int(
                        firstNode.x)-70, y=int(firstNode.y)+70, originalId=nodeOrig.identifier)
                    inputNode.w = 240
                    inputNode.h = 36
                    inputNode.color = nodeOrig.color
                    inputNode.definition = 'result = ' + \
                        str(nodeOrig.identifier)

                    res.append(inputNode.identifier)
        return res

    def createAlias(self, nodeList):
        """Create node alias"""
        res = []
        if not nodeList is None:
            for nodeId in nodeList:
                if self.existNode(nodeId):
                    firstNode = self.getNode(nodeId)
                    nodeOrig = self.getOriginalNode(nodeId)
                    aliasNode = self.createNode(moduleId=firstNode.moduleId, nodeClass='alias', x=int(
                        firstNode.x)+30, y=int(firstNode.y)+30, originalId=nodeOrig.identifier)
                    aliasNode.w = int(nodeOrig.w)
                    aliasNode.h = int(nodeOrig.h)
                    aliasNode.definition = 'result = ' + \
                        str(nodeOrig.identifier)
                    aliasNode.color = BaseNode.getDefaultColor(
                        nodeOrig.nodeClass) if nodeOrig.color is None else nodeOrig.color

                    res.append(aliasNode.identifier)
        return res

    def getOriginalNode(self, nodeId):
        """Get original node from an alias or input node"""
        if self.existNode(nodeId):
            nodeOrig = self.getNode(nodeId)
            if nodeOrig.originalId is None:
                return nodeOrig
            else:
                return self.getOriginalNode(nodeOrig.originalId)

    def isCalcNodes(self, nodeList):
        """Return list of Booleans. True for node is calculated otherwise False"""
        res = []
        if not nodeList is None:
            for nodeId in nodeList:
                isCalc = False
                if self.existNode(nodeId):
                    isCalc = self.getNode(nodeId).isCalc
                res.append(isCalc)
        return res

    def exportModule(self, moduleId, filename):
        """Export module to file"""
        _moduleIOEngine = IOModule(self)
        return _moduleIOEngine.exportModule(moduleId, filename)

    def importModule(self, moduleId, filename, importType):
        """Import module from file"""
        _moduleIOEngine = IOModule(self)
        return _moduleIOEngine.importModule(moduleId, filename, importType)

    def exportFlatNode(self, nodeId, numberFormat, columnFormat, fileName):
        """Export node values in flat format"""
        if self.existNode(nodeId):
            evaluator = Evaluator.createInstance(self.nodeDic[nodeId].result)
            return evaluator.exportFlatNode(self.nodeDic, nodeId, numberFormat, columnFormat, fileName)
        return False

    def dumpNodeToFile(self, nodeId, fileName):
        """Dump current node value to file"""
        if self.existNode(nodeId):
            evaluator = Evaluator.createInstance(self.nodeDic[nodeId].result)
            return evaluator.dumpNodeToFile(self.nodeDic, nodeId, fileName)
        return False

    def saveModel(self, fileName=None):
        """Save model. If fileName is specified, then save to fileName, else return string of ppl """

        # update used libraries
        self.modelProp['libs'] = self._get_used_libraries()

        toSave = {
            'modelProp': self.modelProp,
            'nodeList': []
        }

        for k, v in self.nodeDic.items():
            if(not v.system):
                toSave['nodeList'].append(v.toObj())

        if fileName:
            # Save existing model
            if os.path.isfile(fileName):
                filename_to_save = f'{fileName}.tmp#'
                filename_old = f'{fileName}.old#'

                # remove old .tmp file
                if os.path.isfile(filename_to_save):
                    os.remove(filename_to_save)
                # remove old .old file
                if os.path.isfile(filename_old):
                    os.remove(filename_old)

                with open(filename_to_save, 'w') as f:
                    f.write(jsonpickle.encode(toSave))
                    f.close()

                # move old ppl to filename_old
                os.rename(fileName, filename_old)
                # move filename_to_save to ppl
                os.rename(filename_to_save, fileName)

                if os.path.isfile(filename_old):
                    new_size = os.path.getsize(fileName)
                    old_size = os.path.getsize(filename_old)
                    if new_size/old_size > 0.8:
                        # remove old if new file is greater
                        os.remove(filename_old)
                    else:
                        os.rename(
                            filename_old, f"{fileName}-{datetime.datetime.today().strftime('%Y%m%d-%H%M%S')}.old")

            # Save new model
            else:
                with open(fileName, 'w') as f:
                    f.write(jsonpickle.encode(toSave))
                    f.close()
        return jsonpickle.encode(toSave)

    def openModel(self, fileName=None, textModel=None):
        """Open model.
        If fileName is especified then open from fileName, else open from textModel text """

        self.release()

        opened = {}
        if textModel:
            opened = jsonpickle.decode(textModel)
        else:
            if not self.isLinux():
                fileName = fileName.replace('/', '\\')
            with open(fileName, 'r') as f:
                opened = jsonpickle.decode(f.read())
                f.close()

        self._modelProp = opened['modelProp']

        self._isLoadingModel = True

        # create base model node
        multiplier = 1
        hasBaseNode = False
        for obj in opened['nodeList']:
            if obj['moduleId'] == '_model_':
                node = self.createNode(obj['identifier'], moduleId='_model_')
                # multiplier for old models
                if obj['w'] == 50 and obj['h'] == 25:
                    # the model is an old model
                    multiplier = 2
                obj['w'] = str(int(obj['w']) * multiplier)
                obj['h'] = str(int(obj['h']) * multiplier)
                node.fromObj(obj)
                self._modelNode = node
                hasBaseNode = True
                break

        if not hasBaseNode:
            self.initialize()

        # creating system nodes
        self.createSystemNodes(fileName)

        rootModelId = self.modelNode.identifier

        # create nodes
        for obj in opened['nodeList']:
            if obj['moduleId'] != '_model_' and obj['identifier']:
                if obj['nodeClass'] in ['alias', 'formnode']:
                    if hasattr(obj, 'originalId'):
                        node = self.createNode(
                            obj['identifier'], moduleId=rootModelId, originalId=obj['originalId'])
                    else:
                        index = obj['definition'].find('=')
                        originalId = obj['definition'][index+1:].strip()
                        node = self.createNode(
                            obj['identifier'], moduleId=rootModelId, originalId=originalId)
                else:
                    node = self.createNode(
                        obj['identifier'], moduleId=rootModelId)

                obj['w'] = str(int(obj['w']) * multiplier)
                obj['h'] = str(int(obj['h']) * multiplier)
                node.fromObj(obj)
                node = None

        [self.nodeDic[nod].generateIO() for nod in self.nodeDic]

        opened = None
        self._isLoadingModel = False

        # check models library
        self.ensureModelLibraries()

        # apply backward compatibility
        self.applyBackwardCompatibility()

        # evaluate nodes on start
        try:
            for key in self.nodeDic:
                if self.nodeDic[key] and self.nodeDic[key].evaluateOnStart:
                    _dummy = self.nodeDic[key].result
        except Exception as ex:
            print(str(ex))
            # TODO: send via channels msg to client
            pass

        return True

    def closeModel(self):
        """Close model"""
        self.release()

    def getCustomImports(self):
        """Return object with custom imported python modules."""
        if self._customImports is None:
            self._buildCustomImports()
        return self._customImports

    def _buildCustomImports(self):
        """Build custom imports"""

        self._customImports = Model.DEFAULT_IMPORTS.copy()

        # add pyplan funcions
        self._customImports["pp"] = PyplanFunctions(self)
        # for backward compatibility
        self._customImports["selector"] = self._customImports["pp"].selector

        # support old 'default imports' node
        if self.existNode('imports'):
            import_node = self.getNode('imports')
            import_dic = import_node.result
            for key in import_dic:
                if not key in self._customImports:
                    self._customImports[key] = import_dic[key]

    def createSystemNodes(self, fileName):
        """Create system nodes"""
        # current path
        systemPathNode = self.createNode(
            identifier='current_path', moduleId=self.modelNode.identifier)

        path = os.path.abspath(
            fileName[0:fileName.rfind(os.path.sep)]) + os.path.sep
        if self.isLinux():
            path = fileName[:fileName.rfind('/')] + '/'
            self.createSymlinks(path)
        else:
            path = path.replace('\\', '\\\\')

        systemPathNode.system = True
        systemPathNode.definition = 'result="""' + str(path) + '"""'
        os.chdir(str(path))

        node = self.createNode(identifier='pyplan_user',
                               moduleId=self.modelNode.identifier)
        node.system = True

        node = self.createNode(identifier='cub_refresh',
                               moduleId=self.modelNode.identifier)
        node.system = True
        node = self.createNode(identifier='pyplan_refresh',
                               moduleId=self.modelNode.identifier)
        node.system = True

        node = self.createNode(
            identifier='_scenario_', moduleId=self.modelNode.identifier, nodeClass='index')
        node.system = True
        node.title = 'Scenario'
        node.definition = "result = pp.index(['Current'])"

        node = self.createNode(identifier='task_log_endpoint',
                               moduleId=self.modelNode.identifier, nodeClass='variable')
        node.system = True
        node.title = 'TaskLog endpoint'
        node.definition = "result = ''"

    def createSymlinks(self, fileName):
        if os.getenv('PYPLAN_IDE', '0') == '1' or os.getenv('ENGINE_MODE', '') in ['fixed', 'local']:
            return

        # Add user or public path to system paths
        path_sep = os.path.sep
        path_arr = fileName[:fileName.rfind(path_sep)].split(path_sep)

        dest_path = path_sep.join(
            path_arr[:6 if path_arr[4] == 'Teams' else 5])

        # Get python folder path
        python_folder = f'python{sys.version[:3]}'
        try:
            folder_list = os.listdir(os.path.join(dest_path, '.venv', 'lib'))
            python_folder = folder_list[len(folder_list)-1]
        except Exception as ex:
            pass

        # Add user/public library to system paths
        user_lib_path = os.path.join(
            dest_path, '.venv', 'lib', python_folder, 'site-packages')
        venv_path = os.path.join('/venv', 'lib', 'python3.7', 'site-packages')

        try:
            if not os.path.isdir(user_lib_path):
                os.makedirs(user_lib_path, exist_ok=True)
            # copy base venv folders
            os.system(f'cp -r -u {venv_path}-bkp/* "{user_lib_path}"')
        except:
            pass

        # create symlink from user /public site-package
        if os.path.isdir(user_lib_path):
            os.system(f'rm -rf {venv_path}')
            os.system(f'ln -s -f "{user_lib_path}" {venv_path}')

    def applyBackwardCompatibility(self):
        # update old selector definition
        if self.existNode("selector"):
            node = self.getNode("selector")
            if node.nodeClass == "function" and "from pyplan_engine.classes.PyplanFunctions import Selector" in node.definition:
                node.definition = "result = pp.selector"

        # check for cubepy in imports node
        if self.existNode("imports"):
            node = self.getNode("imports")
            if "cubepy" in node.definition and not "cubepy" in sys.modules:
                sys.modules["cubepy"] = cubepy

    def isLinux(self):
        if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
            return True
        else:
            return False

    # profileNode
    def profileNode(self, nodeId):
        """Perform profile of an node"""
        if self.getNode(nodeId).originalId is not None:
            nodeId = self.getNode(nodeId).originalId
        profile = self.getNode(nodeId).profileNode(
            [], [], self.getNode(nodeId).evaluationVersion, nodeId)

        for node in profile:
            node['calcTime'] = node['evaluationTime'] if node['evaluationTime'] > 0 else 0

        # Fix acumulated time
        total_time = 0
        for nn in reversed(range(len(profile))):
            node = profile[nn]
            total_time = total_time + node['calcTime']
            node['evaluationTime'] = total_time

        return jsonpickle.encode(profile)

    def evaluate(self, definition, params=None, returnEvaluateTime=False):
        """Evaluate expression"""
        res = None
        evalNode = BaseNode(
            model=self, identifier='__evalnode__', nodeClass='variable')
        evalNode._definition = definition
        evalNode.calculate(params)
        res = evalNode.result
        evaluateTime = evalNode.lastEvaluationTime
        evalNode.release()
        evalNode = None
        if returnEvaluateTime:
            return res, evaluateTime
        return res

    def callFunction(self, nodeId, params=None):
        """Call node function with params"""
        res = None
        if self.existNode(nodeId):
            nodeFn = self.getNode(nodeId).result
            if params is None:
                res = nodeFn()
            else:
                res = nodeFn(**params)
        return res

    def getIdentifierByNode(self, result):
        """Return Identifier of node searching by your result"""
        res = ''
        for nodeId in self.nodeDic:
            if self.nodeDic[nodeId].isCalc:
                if self.nodeDic[nodeId].result is result:
                    res = nodeId
                    break
        return res

    def loadScenario(self, nodeId, scenarioData):
        """ Load and fill scenarioDic """
        res = False
        if not scenarioData is None:
            if self.existNode(nodeId):

                scenarioResult = []
                scenarioNames = []

                scenList = str(scenarioData).split('##')
                for scenItem in scenList:
                    arr = str(scenItem).split('||')
                    if len(arr) == 3:
                        scenarioName = arr[1]
                        fileName = arr[2]
                        _result = None
                        if arr[0] == '-1':  # current
                            _result = self.getNode(nodeId).result
                        else:
                            nodeDef = ''
                            with open(fileName, 'r') as f:
                                nodeDef = f.read()
                                f.close()
                            _result = self.evaluate(nodeDef)
                        scenarioResult.append(_result)
                        scenarioNames.append(scenarioName)

                # fill scenario node
                scenarioIndex = self.getNode('_scenario_')
                scenarioIndex.definition = "result = pp.index(['" + "','".join(
                    scenarioNames) + "'])"

                if len(scenarioResult) > 0:
                    finalResult = None

                    if isinstance(scenarioResult[0], xr.DataArray):
                        finalResult = xr.concat(
                            scenarioResult, scenarioIndex.result.dataArray)
                    else:
                        finalResult = xr.DataArray(
                            scenarioResult, scenarioIndex.result.coord)

                    self._scenarioDic[nodeId] = finalResult
                    res = True

        return res

    def unloadScenario(self):
        """Unload all scenarios and clean calculated nodes in the scenario """
        for key in self._scenarioDic:
            self._scenarioDic[key] = None

        self._scenarioDic = dict()
        scenarioIndex = self.getNode('_scenario_')
        scenarioIndex.definition = "result = pp.index(['Current'])"

    def geoUnclusterData(self, nodeId, rowIndex, attIndex, latField='latitude', lngField='longitude', geoField='geoField', labelField='labelField', sizeField='sizeField', colorField='colorField', iconField='iconField'):
        """get unclusted data for geo representation."""
        if self.existNode(nodeId):
            result = self.nodeDic[nodeId].result
            if not result is None:
                self.evaluationVersion += 1
                evaluator = Evaluator.createInstance(result)
                return evaluator.geoUnclusterData(result, self.nodeDic, nodeId, rowIndex, attIndex, latField, lngField, geoField, labelField, sizeField, colorField, iconField)
        else:
            return ''

    def callWizard(self, param):
        """ Start and call wizard toolbar """
        obj = jsonpickle.decode(param)
        key = obj['wizard']
        action = obj['action']
        params = obj['params']

        if self._wizard is None or self._wizard.code != key:
            self._wizard = self._getWizzard(key)

        toCall = getattr(self._wizard, action)

        return toCall(self, params)

    def createNewModel(self, modelFile=None, modelName=None):
        self.release()
        self._modelProp = {}
        self._isLoadingModel = True
        self.initialize(modelName)
        self.createSystemNodes(modelFile)
        self._isLoadingModel = False
        if modelFile:
            self.saveModel(modelFile)
        return True

    def _getWizzard(self, key):
        key = key.lower()
        if key == 'calculatedfield':
            return CalculatedField.Wizard()
        elif key == 'selectcolumns':
            return SelectColumns.Wizard()
        elif key == 'selectrows':
            return SelectRows.Wizard()
        elif key == 'sourcecsv':
            return sourcecsv.Wizard()
        elif key == 'dataframeindex':
            return DataframeIndex.Wizard()
        elif key == 'dataframegroupby':
            return DataframeGroupby.Wizard()
        elif key == 'dataarrayfrompandas':
            return DataarrayFromPandas.Wizard()
        elif key == 'inputtable':
            return InputTable.Wizard()
        elif key == 'createindex':
            return CreateIndex.Wizard()
        elif key == 'indexfrompandas':
            return IndexFromPandas.Wizard()
        elif key == 'dataarrayfilter':
            return DataArrayFilter.Wizard()
        elif key == 'editindex':
            return EditIndex.Wizard()
        elif key == 'renameindexitem':
            return RenameIndexItem.Wizard()

    def getSystemResources(self, onlyMemory=False):
        """Return current system resources"""

        def _read_int(file):
            data = 0
            with open(file, 'r') as f:
                data = int(f.read())
                f.close()
            return data

        def _read_cache():
            data = 0
            with open('/sys/fs/cgroup/memory/memory.stat', 'r') as f:
                line = f.readline()
                data = int(str(line).split(' ')[1])
                f.close()
            return data

        mem_limit = _read_int(
            '/sys/fs/cgroup/memory/memory.limit_in_bytes') / 1024/1024/1024
        if mem_limit > 200:  # bug for container whitout limit
            total_host = ''
            with open('/proc/meminfo', 'r') as f:
                line1 = f.readline()
                total_host = str(line1).split(' ')[-2:-1][0]
                mem_limit = int(total_host) / 1024/1024

        mem_used = (_read_int(
            '/sys/fs/cgroup/memory/memory.usage_in_bytes') - _read_cache()) / 1024/1024/1024

        max_mem = _read_int(
            '/sys/fs/cgroup/memory/memory.max_usage_in_bytes') / 1024/1024/1024

        # get cpu usage

        if onlyMemory:
            return {
                'totalMemory': mem_limit,
                'usedMemory': mem_used,
                "maxMemory": max_mem
            }
        else:

            time_1 = datetime.datetime.now()
            cpu_time_1 = _read_int('/sys/fs/cgroup/cpu/cpuacct.usage')
            sleep(0.3)
            time_2 = datetime.datetime.now()
            cpu_time_2 = _read_int('/sys/fs/cgroup/cpu/cpuacct.usage')
            delta_time = (time_2 - time_1).microseconds * 10
            used_cpu = (cpu_time_2 - cpu_time_1) / delta_time
            used_cpu = used_cpu if used_cpu < 100 else 100

            current_node = self._currentProcessingNode
            if self.existNode(current_node):
                node = self.getNode(current_node)
                if node.title:
                    current_node = f'{node.title} ({current_node})'

            return {
                'totalMemory': mem_limit,
                'usedMemory': mem_used,
                'usedCPU': used_cpu,
                'pid': self.getPID(),
                'currentNode': current_node,
                "maxMemory": max_mem
            }

    def ensureModelLibraries(self):
        """Ensure that all model libs are installed"""
        try:
            if not 'libs' in self.modelProp:
                self.modelProp['libs'] = []
            modelLibs = self.modelProp['libs']
            # get current installed libs
            installed_libs_str = self.listInstalledLibraries()
            installed_libs = jsonpickle.decode(installed_libs_str)
            installed_libs_dic = dict()
            for installed_lib in installed_libs:
                installed_libs_dic[installed_lib['name']
                                   ] = installed_lib['version']

            to_install = ''
            for lib in modelLibs:
                if not lib['name'] in installed_libs_dic:
                    to_install += f" {lib['name']}=={lib['version']}"
                # TODO: if exists, check version and send message via channels

            if to_install != '':
                self._installLibrary(to_install)

        except Exception as ex:
            print(f'Error checking libraries. {ex}')
            # TODO: send to client via channels

    def _installLibrary(self, lib):

        cmd = f'pip install {lib}'

        # If there are proxy configurations, use them to install from pip
        http_proxy = os.getenv('PYPLAN_HTTP_PROXY')
        https_proxy = os.getenv('PYPLAN_HTTPS_PROXY')
        if http_proxy:
            cmd += f' --proxy {http_proxy}'
        elif https_proxy:
            cmd += f' --proxy {https_proxy}'

        p = subprocess.Popen(split(cmd), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
        nn = 0
        while p.stdout is not None and nn < 1200:
            # TODO: show feedback to ide using channels
            line = p.stdout.readline()
            if not line:
                p.stdout.flush()
                aa, err = p.communicate()
                if err:
                    self._currentInstallProgress.append(err.rstrip('\n'))
                break
            sleep(1)
            nn += 1
            self._currentInstallProgress.append(line.rstrip('\n'))

        importlib.invalidate_caches()

    def installLibrary(self, lib, target):
        """install python library"""
        self._currentInstallProgress = []
        thread = threading.Thread(target=self._installLibrary, args=(lib,))
        thread.start()
        return 'ok'

    def _get_used_libraries(self):
        try:
            _regex = re.compile(r'(^import\s.*)|(^from\s.*)', re.M)
            _temp_imports = []
            _imports = dict()
            _used_libraries = []

            # retrieve all import statements
            for node_id in self.nodeDic:
                _def = self.nodeDic[node_id].definition
                if _def:
                    _finditer = re.finditer(_regex, _def)
                    for match in _finditer:
                        if match.group() not in _temp_imports:
                            _temp_imports.append(match.group())

            # filter imports (make distinct, check type)
            # name == pypi name
            for _element in _temp_imports:
                if _element[:6] == 'import':
                    # import
                    if _element[7:].find(',') != -1:
                        # multiple imports in one line
                        for _el in _element[7:].split(','):
                            _imports[self._check_import_function(_el.strip())] = {'import_name': self._check_import_function(
                                _el.strip()), 'import_type': 'import', 'name': None, 'version': None}
                    elif _element[7:].find(' as ') != -1:
                        # import has an alias
                        _el = _element[7:_element.find(' as ')].strip()
                        _imports[self._check_import_function(_el)] = {'import_name': self._check_import_function(
                            _el), 'import_type': 'import', 'name': None, 'version': None}
                    else:
                        # full import
                        _el = _element[7:].strip()
                        _imports[self._check_import_function(_el)] = {'import_name': self._check_import_function(
                            _el), 'import_type': 'import', 'name': None, 'version': None}
                elif _element[:4] == 'from':
                    # from
                    if _element[5:].find(' import ') != -1:
                        _el = _element[5:_element.find(' import ')].strip()
                        _imports[self._check_import_function(_el)] = {'import_name': self._check_import_function(
                            _el), 'import_type': 'from', 'name': None, 'version': None}

            _installed_libs = self._check_installed_libraries()
            for key in _imports.keys():
                for lib in _installed_libs:
                    if lib == _imports[key]['import_name']:
                        _imports[key].update(_installed_libs[lib])
                if _imports[key]['name'] != None:
                    _used_libraries.append(_imports[key])

            return _used_libraries
        except Exception as ex:
            if self.ws:
                self.ws.sendMsg(str(ex), 'Error getting used libraries',
                                not_level=ws_settings.NOTIFICATION_LEVEL_ERROR)
            return []

    def _check_import_function(self, _element):
        if _element.find('.') != -1:
            # the element has a function been imported
            return _element[:_element.find('.')].strip()
        else:
            return _element

    def _check_installed_libraries(self):
        try:
            installed_libraries = dict()
            dirs = sys.path if os.getenv('PYPLAN_IDE') else [
                '/venv/lib64/python3.7/site-packages/']
            for _dir in dirs:
                installed_modules = dict()
                if '.zip' not in _dir:
                    for folder in os.listdir(_dir):
                        # get dist.info folders
                        if '.dist-info' in folder or '.egg-info' in folder:
                            top_level_file = os.path.join(
                                _dir, folder, 'top_level.txt')
                            if os.path.isfile(top_level_file):
                                top_level = ""
                                try:
                                    top_level = str(
                                        open(top_level_file).read()).replace('\n', '')
                                except Exception as ex:
                                    top_level = str(
                                        open(top_level_file, encoding="utf8").read()).replace('\n', '')
                                metadata_file = os.path.join(
                                    _dir, folder, 'METADATA' if '.dist-info' in folder else 'PKG-INFO')
                                if os.path.isfile(metadata_file):
                                    metadata = ""
                                    try:
                                        metadata = str(
                                            open(metadata_file).read())
                                    except Exception as ex:
                                        metadata = str(
                                            open(metadata_file, encoding="utf8").read())
                                    metadata_arr = metadata.split('\n')
                                    for metadata_item in metadata_arr:
                                        if str(metadata_item).startswith('Name: '):
                                            pypi_name = str(
                                                metadata_item).split(' ')[1]
                                        elif str(metadata_item).startswith('Version: '):
                                            pypi_version = str(
                                                metadata_item).split(' ')[1]
                                installed_modules[top_level] = {
                                    'name': pypi_name, 'version': pypi_version}
                installed_libraries.update(installed_modules)
            return installed_libraries
        except Exception as ex:
            if self.ws:
                self.ws.sendMsg(str(ex), 'Error checking installed libraries',
                                not_level=ws_settings.NOTIFICATION_LEVEL_ERROR)
            return {}

    def listInstalledLibraries(self):
        cmd = 'pip list -v --disable-pip-version-check --format=json'
        popen = subprocess.Popen(
            split(cmd), stdout=subprocess.PIPE, universal_newlines=True)

        stdout, stderr = popen.communicate()
        if stderr:
            raise ValueError(
                f'Error listing installed libraries: {str(stderr)}')

        return stdout

    def uninstallLibrary(self, lib, target):
        """Uninstall python library"""
        # cmd = f'pip uninstall -y {lib}'
        # We cant use pip uninstall so we do a dirty workaround
        cmd = f"find {target} -name '*{lib.replace('-','_')}*' -exec rm -rf {{}} \;"
        popen = subprocess.Popen(
            split(cmd), stdout=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = popen.communicate()
        importlib.invalidate_caches()
        if stderr:
            raise ValueError(f'Error uninstalling library: {str(stderr)}')

        return stdout

    def getInstallProgress(self, from_line):
        """Return install library progress"""
        from_line = int(from_line)
        if len(self._currentInstallProgress) == 0 or from_line > len(self._currentInstallProgress):
            return []
        else:
            return self._currentInstallProgress[from_line:]

    def setNodeIdFromTitle(self, node_id):
        """Generate node id from node identifier """
        res = {'node_id': node_id}
        model_props = self.getModelProperties()
        if (not 'changeIdentifier' in model_props or model_props['changeIdentifier'] == '1') and self.existNode(node_id):
            node = self.nodeDic[node_id]
            new_id = ''
            try:
                if node.title:
                    new_id = self._removeDiacritics(node.title)
            except Exception as ex:
                raise ValueError(f'Error finding node title: {str(ex)}')

            if new_id:
                if self.existNode(new_id):
                    new_id = self.getNextIdentifier(new_id)
                node.identifier = new_id
                res['node_id'] = new_id
        return res

    def _removeDiacritics(self, text):
        """Removes all diacritic marks from the given string"""
        norm_txt = unicodedata.normalize('NFD', text)
        shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
        # remove accents and other diacritics, replace spaces with '_' because identifiers can't have spaces
        no_spaces = unicodedata.normalize(
            'NFC', shaved).lower().replace(' ', '_')
        final_text = no_spaces
        # only allow [a-z], [0-9] and _
        p = re.compile('[a-z0-9_]+')
        for i in range(0, len(no_spaces)):
            if not (p.match(no_spaces[i])):
                final_text = final_text[:i] + '_' + final_text[i+1:]
        # i the first char is not a-z then replaceit (all identifiers must start with a letter)
        p2 = re.compile('[a-z]+')
        if not p2.match(final_text[0]):
            final_text = 'a' + final_text[1:]
        return final_text
