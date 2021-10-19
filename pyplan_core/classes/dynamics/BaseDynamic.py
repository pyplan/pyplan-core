import re


class BaseDynamic(object):

    @staticmethod
    def clearAllCircularDependency(definition: str) -> str:
        """ Replaces cp.dynamic(x,y,z) , pp.dynamic(x,y,z) and dynamic(x,y,z) for its initialValues"""
        new_def = definition
        replacement_value = '0'

        dynamic_positions = BaseDynamic.getPositionsFromDefinition(definition)
        if dynamic_positions is not None:
            dynamic_initial_pos, dynamic_last_pos = dynamic_positions
            dynamic_def = definition[dynamic_initial_pos:dynamic_last_pos+1]

            # Get initialValues parameter
            dynamic_params = BaseDynamic.getParametersFromDefinition(
                definition)
            if dynamic_params is not None and len(dynamic_params) >= 4:
                initial_values_param = dynamic_params[3]
                # Remove 'initialValues='
                if initial_values_param.startswith('initialValues='):
                    initial_values_param = initial_values_param.replace(
                        'initialValues=', '')
                replacement_value = initial_values_param

            new_def = definition.replace(dynamic_def, replacement_value)

        return new_def

    @staticmethod
    def getPositionsFromDefinition(definition: str) -> tuple:
        """Returns tuple with start and end positions of first occurrence of:
                - dynamic()
                - pp.dynamic()
                - cp.dynamic()
        """
        pattern = r'(pp.|cp.)?dynamic\s*\('
        dynamic_search = re.search(pattern, definition)
        if dynamic_search is not None:
            dynamic_initial_pos = dynamic_search.span()[0]
            new_def = definition[dynamic_initial_pos:]

            final_pos = -1
            open_prt = 0
            # Stop counting when number of open parentheses matches number of closing parentheses
            for pos, word in enumerate(new_def, dynamic_initial_pos):
                if word == '(':
                    open_prt += 1
                elif word == ')':
                    open_prt -= 1
                    if open_prt == 0:
                        final_pos = pos
                        break

            if final_pos != -1:
                return dynamic_initial_pos, final_pos

    @staticmethod
    def getParametersFromDefinition(definition: str) -> list:
        """Returns a list of string parameters of first dynamic function in definition"""
        params_pattern = r',(?![^(]*\))'

        dynamic_positions = BaseDynamic.getPositionsFromDefinition(definition)
        if dynamic_positions is not None:
            dynamic_initial_pos, dynamic_last_pos = dynamic_positions
            dynamic_def = definition[dynamic_initial_pos:
                                     dynamic_last_pos+1].replace(' ', '')
            first_prt_pos = dynamic_def.find('(')
            # remove (pp.|cp.)dynamic( and last parenthesis
            dynamic_params_def = dynamic_def[first_prt_pos+1:
                                             len(dynamic_def)-1]

            return re.split(params_pattern, dynamic_params_def)


# implementations


    def circularEval(self, node, params): raise ValueError("not implemented")

    def generateLoopDef(self, node, nodeDefinition,
                        cyclicVariables): raise ValueError("not implemented")

    def generateInitDef(self, node, nodeResult,
                        dynamicIndex): raise ValueError("not implemented")

    def generateCircularParameters(
        self, node, nodeList): raise ValueError("not implemented")

    def clearCircularDependency(
        self, stringDef, replaceWith="0"): raise ValueError("not implemented")
