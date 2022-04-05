# Python

- `printing with leading zeros`

Sometimes you want a log line to have always the same length, for easy readability. In such a case you can easily use python string formatting to pad the int with leading zeros, to the specified length:

```
var = 35
print(f'{var:04d}')
# prints 0035 (pads to 4 digits)

# you can also use zfill()
var_2 = str(var).zfill(4)
print(var_2)
# prints 0035 too
```

- `counting occurrences of each item in a list`

We can use the `collections.Counter()` item easily:

```
l = ['a', 'a', 'b', 'c', 'c', 'c']
from collections import Counter
print(Counter(l))
# will show Counter({'c': 3, 'a': 2, 'b': 1})
```

- `removing all occurrences of an element from a list`

Note that the `list.remove()` function only removes a single occurrence.
So instead we have 2 options:

```
a_list = [2, 0, 3, 5, 0, 0, 1]
value_to_remove = 0

# 1) use a list comprehension
r1 = [e for e in a_list if e != value_to_remove]
print(r1)

# 2) use a filter() function, which takes another function and an iterable
r2 = list(filter((value_to_remove).__ne__, a_list))
print(r2)
```

Here `.__ne__` refers to the _not_equal_ function of the element we want to remove. We could have also used a lambda. The `filter()` function removes all the elements for whom the not-equal will return `True`.

- `installing a specific version of a package`

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
