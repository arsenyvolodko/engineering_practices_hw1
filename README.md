# engineering_practices_hw1

### Installing package manager
`pip install poetry`

### Activating virtual environment
`poetry shell`

### Installing dependencies
`poetry install --no-root`

### Now you can run the code. 
For example, running\
`poetry run python3 some_code_1.py`\
will print some metrics describing models learned on avito estate dataset (file aparts.csv).

### Linters and formatters
You can use linting and formatting tools to check and improve your code.\
For example, `black --check --diff file_name.py` will show you the difference between your code and the code that black formatter would produce. 
You can also use `black file_name.py` to format your code. The same applies to `isort` formatter.
