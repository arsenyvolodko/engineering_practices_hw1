# engineering_practices_hw1

black --check --diff some_code_1.py > LINTING.md
black --check --diff some_code_2.py >> LINTING.md

isort --diff some_code_1.py >> LINTING.md
isort --diff some_code_2.py >> LINTING.md

black some_code_1.py some_code_2.py
isort some_code_1.py some_code_2.py


```