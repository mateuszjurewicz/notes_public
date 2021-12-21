---
tags: [Notebooks/Skills/Tech]
title: Latex
created: '2021-12-21T09:55:45.925Z'
modified: '2021-12-21T09:58:02.299Z'
---

# Latex

Basic notes on $LaTeX$, which in Notable is actual $KaTeX$.

### Figure sizes

Require the use of the `graphicx` package, you can remove the `keepaspectratio` parameter for the figure to fully obey the width and height as specified.
```
\usepackage{graphicx}

\begin{figure}[!t]
    \centering
    \includegraphics[
      width=9cm,
      height=6cm,
      keepaspectratio
  ]
{"images/figure_placeholder.png"}
\hfill
\caption{Example of a set-to-sequence neural task and model.}
\label{fig:placeholder_2}
\end{figure}
```
