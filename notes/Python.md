---
tags: [Notebooks/Skills/Tech]
title: Python
created: '2021-12-07T14:57:32.374Z'
modified: '2021-12-07T15:00:30.807Z'
---

# Python

- `ast` library
for converting nested lists stored as strings in a csv file back into lists:

```
import ast

nested_list_as_str = '[[1, 2, 3],[4, 5, 6],[7, 8, 9]]'

nested_list = ast.literal_eval(nested_list_as_str)
print(len(nested_list), nested_list)
# 3 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

```
