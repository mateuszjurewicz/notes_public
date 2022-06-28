## Flatten a list of lists
Sometimes you have a a list of lists and instead want a flat, 1-dimensional list. This can be achieved by a list comprehension, but not quite like you'd think:

```
nested = [[1, 2, 3], [4, 5, 6], [7, 8]]
flat = [elem for sublist in nested for elem in sublist]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8]
```

You can also use `itertools.chain()`, but the above is cleaner. 

```
import itertools

nested = [[1, 2, 3], [4, 5, 6], [7, 8]]
flat = list(itertools.chain(*nested))
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8]
```

**Note that this won't work with deeper and inconsistent nesting levels**.

## Virutalenv
Can be easily use to set up virtual environments for python installations and specific packages, whenever conda fails.

To create a new virtual environment:

```
python3 -m venv /path/to/new/virtual/environment
```

## Installing a specific version of a package

```
python3 -m pip install torch==1.10.0  # upgrades to specified version
python3 -m pip install --upgrade torch  # upgrade to latest version
```

## Uninstalling a specific package with pip

```
python3 -m pip uninstall <package_name>
```

## Printing with alignment and decimals
Sometimes we want to combine multiple formatting rules, e.g. only show up to a specific decimal point but also left-align, filling with 10 spaces etc. We can do this but the syntax is tricky, so here's an easily reusable example:

```
As = [10, 2, 333, 4.044]
Bs = ['first', 'second', 'third', 'fourth']
Cs = [True, False, True, False]

text = 4.2
for i, a in enumerate(As):
    b = Bs[i]
    c = Cs[i]
    print('{0:<10.2f} | {1:<10} | {2:<10}'.format(a, b, c))
```

 This will print:
 ```
10.00      | first      | 1         
2.00       | second     | 0         
333.00     | third      | 1         
4.04       | fourth     | 0
```

Alternatively we could use:
```
print(f'{a:<10.2f} | {b:<10} | {c:<10}')
```

In the first `'{0:<10.2f}'.format(a)` the `0` specifies that of all arguments passed to format(), we're now dealing with the zeroth one, the `:<10` means it will be left-aligned, leaving 10 characters empty if they're not filled by `a`, and finally `.2f` means we'll print 2 decimal points. The order matters.

Here's a **cheatsheet about proper order**:

```
format_spec     ::=  [[fill]align][sign][#][0][width][grouping_option][.precision][type]
fill            ::=  <any character>
align           ::=  "<" | ">" | "=" | "^"
sign            ::=  "+" | "-" | " "
width           ::=  digit+
grouping_option ::=  "_" | ","
precision       ::=  digit+
type            ::=  "b" | "c" | "d" | "e" | "E" | "f" | "F" | "g" | "G" | "n" | "o" | "s" | "x" | "X" | "%"
```
Source for the above is [here in the docs](https://docs.python.org/3/library/string.html#formatspec)

## Pyplot and garbage collection

Apparently opened plots can sometimes not get properly garbarge collected (memory leak?). Got this warning on GPU: 

```
RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory"
```

The only safe solution seems to be to call `plt.close('all')` at the end of the visualization functions, after the figure gets saved somewhere.

Source [here](https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib).

## Printing with leading zeros

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

## Counting occurrences of each item in a list

We can use the `collections.Counter()` item easily:

```
l = ['a', 'a', 'b', 'c', 'c', 'c']
from collections import Counter
print(Counter(l))
# will show Counter({'c': 3, 'a': 2, 'b': 1})
```

## Removing all occurrences of an element from a list

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

## `ast` library

for converting nested lists stored as strings in a csv file back into lists:

```
import ast

nested_list_as_str = '[[1, 2, 3],[4, 5, 6],[7, 8, 9]]'

nested_list = ast.literal_eval(nested_list_as_str)
print(len(nested_list), nested_list)
# 3 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

```
