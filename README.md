# ![Pyplan](https://raw.githubusercontent.com/pyplan/pyplan-ide/master/docs/assets/img/logo.png)

**Pyplan** is a Python development environment intended for easily creating and deploying data analytics apps. Different than a Jupyter Notebook, where code is structured as a list of sentences, in Pyplan code is contained in nodes that work as calculation steps, organized in a hierarchical influence diagram. Nodes can be evaluated, and its result inspected through a console output or rendered as a table or graph. 

**Pyplan Core** is a Python library for using Pyplan models in any Python environment as for example a Jupyter Notebook.
It provides methods to open a model, set selectors values and get results of nodes.
This library expands the possibilities of using Pyplan models embedded in your own application.
Models still have to be created using the Pyplan Graphical IDE that you can download and use for free at www.pyplan.org

## Installing and running Pyplan Core

You can install Pyplan Core in your computer following the instructions below. 

```bash
pip install pyplan_core
```
or

```bash
conda config --append channels pyplan
conda config --append channels conda-forge
conda install pyplan-core
```

You can use Pyplan Core in your computer following the instructions below. 

```python
from pyplan_core.pyplan import Pyplan

pyplan = Pyplan()

#model_filename = "path/to/my_model_file.ppl"
model_filename = pyplan.sample_models.use_of_pyplan_core()  # for sample purposes
pyplan.openModel(model_filename)
value = pyplan.getResult("total_of_cases")
print(value)

pyplan.closeModel() 
```
