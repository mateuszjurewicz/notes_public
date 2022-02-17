# Python

- installing a specific version of a package:

```
python3 -m pip install torch==1.10.0  # upgrades to specified version
python3 -m pip install --upgrade torch  # upgrade to latest version
```

- `ast` library
for converting nested lists stored as strings in a csv file back into lists:

```
import ast

nested_list_as_str = '[[1, 2, 3],[4, 5, 6],[7, 8, 9]]'

nested_list = ast.literal_eval(nested_list_as_str)
print(len(nested_list), nested_list)
# 3 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

```
