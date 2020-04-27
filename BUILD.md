# ![Pyplan](https://raw.githubusercontent.com/pyplan/pyplan-ide/master/docs/assets/img/logo.png)



## Develop & test


### Prepare for develop

```bash
python3.7 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure VSCode for debug

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Pytest",
            "type": "python",
            "request": "launch",
            "stopOnEntry": false,
            "pythonPath": "${config:python.pythonPath}",
            "module": "pytest",
            "console": "integratedTerminal",
       }
    ]
}



### Tests
```bash
. venv/bin/activate
pytest tests/. --verbosity=1 --disable-warnings
```
