from itertools import combinations


class ArrowsManager(object):

    EXCLUDED_CLASSES = ['model', 'text']

    def __init__(self, model):
        self.model = model

    def get_dict_of_nodes_in_module(self, module_id, node_classes=None, return_ids=False):
        """
        Returns a dictionary of nodes that are located inside module_id, grouped in three keys:
            - alias
            - module
            - others (rest of node classes except model and text, and excludes system nodes)
        
        Parameters:
            - module_id: str
            - node_classes: str or list of str
            - return_ids: bool
        """
    
        if node_classes is not None and isinstance(node_classes, str):
            node_classes = [node_classes]
        
        results_dict = dict()
        for node in self.model.nodeDic.values():
            if not node.system and node.nodeClass not in self.EXCLUDED_CLASSES:
                node_class = node.nodeClass if node.nodeClass in ['module', 'alias'] else 'others'
                node_module_id = node.moduleId
                if node_module_id == module_id and (node_class in node_classes if node_classes is not None else True):
                    results_dict.setdefault(node_class, []).append(node.identifier if return_ids else node)
        
        return results_dict
    
    def get_all_nodes_in_module(self, module_id, nodes_set=None, return_ids=False):
        """
        Returns a set of nodes that are located in module_id and in module_id`s submodules.
        It contains every node class but model, module and text.
        
        Parameters:
            - module_id: str
            - nodes_set: set
            - return_ids: bool
        """
        
        if nodes_set is None:
            nodes_set = set()
        
        nodes_in_module = self.get_dict_of_nodes_in_module(module_id, return_ids=return_ids)
        nodes_aliases_in_module = nodes_in_module.get('others', []) + nodes_in_module.get('alias', [])
        nodes_set.update(nodes_aliases_in_module)
        
        submodules = nodes_in_module.get('module', [])
        if submodules:
            # Recursively search and add nodes inside child modules
            for submodule in submodules:
                nodes_set = self.get_all_nodes_in_module(
                    module_id=submodule if return_ids else submodule.identifier,
                    nodes_set=nodes_set,
                    return_ids=return_ids)
        
        return nodes_set
    
    def get_node_inputs_outputs_ids(self, node, add_aliases=False, aliases=None, add_inputs=True, add_outputs=True):
        """
        Returns a tuple containing a list of inputs and a list of outputs of a given BaseNode object
        Parameters:
            - node: pyplan_core.classes.BaseNode
            - add_aliases: bool
            - aliases: dict like {node_id: [list of aliases ids]}
            - add_inputs: bool
            - add_outputs: bool
        """
        
        node_inputs_ids = []
        node_outputs_ids = []
        if node.nodeClass == 'alias':
            node_original_id = node.originalId
            if node_original_id in self.model.nodeDic:
                if add_inputs:
                    node_inputs_ids = self.model.nodeDic[node_original_id].inputs
                if add_outputs:
                    node_outputs_ids = self.model.nodeDic[node_original_id].outputs
        else:
            if add_inputs:
                node_inputs_ids = node.inputs
            if add_outputs:
                node_outputs_ids = node.outputs
        
        inputs_ids = set(node_inputs_ids)
        outputs_ids = set(node_outputs_ids)
        
        # Add every alias from the original node as an input or output
        if add_aliases and aliases is not None:
            if add_inputs:
                for input_id in node_inputs_ids:
                    if input_id in aliases:
                        inputs_ids.update(aliases[input_id])
            
            if add_outputs:
                for output_id in node_outputs_ids:
                    if output_id in aliases:
                        outputs_ids.update(aliases[output_id])
        
        return inputs_ids, outputs_ids

    def get_arrows(self, module_id):
        """
        Returns a list of dicts of all arrows inside module_id.
        
        Parameters:
            - module_id: str
        
        Example output: [{`from`: `node_id_1`, `to`: `node_id_2`}, {`from`: `node_id_1`, `to`: `node_id_3`}]
        """
        
        arrows_list = []
        if module_id in self.model.nodeDic:
            nodes = self.get_dict_of_nodes_in_module(module_id)
            aliases = nodes.get('alias', [])
            nodes_in_module = nodes.get('others', []) + aliases
            nodes_ids_in_module = set([nd.identifier for nd in nodes_in_module])
            submodules_in_module = [module for module in nodes.get('module', []) if module.nodeInfo.showInputs or module.nodeInfo.showOutputs]

            aliases_dict = dict()
            for alias in aliases:
                alias_id = alias.identifier
                node_original_id = self.model.nodeDic[alias_id].originalId
                aliases_dict.setdefault(node_original_id, []).append(alias_id)

            # Get every node and inputs inside all submodules recursively
            # This helps searching with node-module and module-module arrows
            all_nodes_and_inputs_ids_by_module = dict()
            for submodule in submodules_in_module:
                submodule_id = submodule.identifier
                all_nodes_and_inputs_ids_by_module[submodule_id] = dict()
                all_nodes_and_inputs_ids_by_module[submodule_id]['nodes'] = self.get_all_nodes_in_module(submodule_id, return_ids=True)
                inputs_ids = set()
                for node in all_nodes_and_inputs_ids_by_module[submodule_id]['nodes']:
                    node_inputs_ids, _ = self.get_node_inputs_outputs_ids(
                        self.model.getNode(node),
                        add_aliases=True,
                        aliases=aliases_dict,
                        add_inputs=True,
                        add_outputs=False)  # outputs not needed since if A is input of B, B is output of A
                    inputs_ids.update(node_inputs_ids)
                all_nodes_and_inputs_ids_by_module[submodule_id]['inputs'] = inputs_ids

            arrows_set = set()
            for node in nodes_in_module:
                if node.nodeInfo.showInputs or node.nodeInfo.showOutputs:
                    node_id = node.identifier
                    # Get node inputs and outputs
                    inputs_ids, outputs_ids = self.get_node_inputs_outputs_ids(
                        node, 
                        add_aliases=True, 
                        aliases=aliases_dict,
                        add_inputs=True,
                        add_outputs=True)
                    
                    # NODE TO NODE
                    if node.nodeInfo.showInputs:
                        for input_id in inputs_ids:
                            if input_id in nodes_ids_in_module and self.model.nodeDic[input_id].nodeInfo.showOutputs:
                                arrows_set.add((input_id, node_id))
                    if node.nodeInfo.showOutputs:
                        for output_id in outputs_ids:
                            if output_id in nodes_ids_in_module and self.model.nodeDic[output_id].nodeInfo.showInputs:
                                arrows_set.add((node_id, output_id))
                    
                    # MODULE TO NODE
                    for submodule in submodules_in_module:
                        submodule_id = submodule.identifier
                        nodes_ids_in_submodule = all_nodes_and_inputs_ids_by_module[submodule_id]['nodes']
                        
                        if node.nodeInfo.showInputs and submodule.nodeInfo.showOutputs:
                            for input_id in inputs_ids:
                                if input_id in nodes_ids_in_submodule:
                                    arrows_set.add((submodule_id, node_id))
                                    break
                        if node.nodeInfo.showOutputs and submodule.nodeInfo.showInputs:
                            for input_id in outputs_ids:
                                if input_id in nodes_ids_in_submodule:
                                    arrows_set.add((node_id, submodule_id))
                                    break

            # MODULE TO MODULE
            for combination in combinations(all_nodes_and_inputs_ids_by_module, 2):
                # Every combination is a pair of (module_1, module_2)
                first_module_id = combination[0]
                first_module = self.model.getNode(first_module_id)
                second_module_id = combination[1]
                second_module = self.model.getNode(second_module_id)
                first_module_nodes_ids = all_nodes_and_inputs_ids_by_module[first_module_id]['nodes']
                second_module_nodes_ids = all_nodes_and_inputs_ids_by_module[second_module_id]['nodes']
                first_module_inputs_ids = all_nodes_and_inputs_ids_by_module[first_module_id]['inputs']
                second_module_inputs_ids = all_nodes_and_inputs_ids_by_module[second_module_id]['inputs']
                
                if first_module.nodeInfo.showOutputs and second_module.nodeInfo.showInputs:
                    for node_id in first_module_nodes_ids:
                        if node_id in second_module_inputs_ids:
                            arrows_set.add((first_module_id, second_module_id))
                            break
                if first_module.nodeInfo.showInputs and second_module.nodeInfo.showOutputs:
                    for node_id in second_module_nodes_ids:
                        if node_id in first_module_inputs_ids:
                            arrows_set.add((second_module_id, first_module_id))
                            break
            
            # Adapt output
            arrows_list = [{'from': arrow[0], 'to': arrow[1]} for arrow in arrows_set]

        return arrows_list