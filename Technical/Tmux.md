# Tmux

https://gist.github.com/MohamedAlaa/2961058

## Kill a detached session

First, to find it's identifier:

`tmux ls`

Then, to kill it:

`tmux kill-session -t your_session_number`

### Move panes
- `ctrl + b + }` 

move the current pane right (or left with `{`). Effectively swaps the placement of panes.

### Scroll in pane

- `ctrl + b + [` 

allows for scrolling in the window (no shift needed for `[`, as opposed to `{`), `ctrl + c` to exit scrolling mode. 


### General
- `ctrl + b + n` - move to next window 
- `ctrl + b + p` - move to previous window
- `ctrl + b + %` - split window vertically
- `ctrl + b + "` - split window horizontally
