# SSH


If you log into a machine using ssh and then your connection breaks, you may want the started script to keep running, to be checked up on after you reconnect. 

## SSH Execution and Broken Connection

### 1. Using Tmux

You can also start tmux on the remote machine, start running whatever script you want via the `python` command, and then detach from the tmux session by using `ctrl + b + d`. It will keep runinng even if I lose the ssh connection used to start tmux.

To reattach that session after logging in again via ssh, you can use `tmux ls` to find its integer number and then run `tmux attach-session -t X`.

Source [here](https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session)

### 2. Using Nohup 
You can do this by starting a tmux session on the remote machine or by executing the original command with the `nohup` (for no-hanging-up) prefix:

```
(venv) user@paperspace1234:~/$ nohup python3 run_experiment.py &
```
Note that you need the ampersand at the bottom. This will result in a running terminal with no output.
To see live updates of the stdout for the running script, you need to run this in a separate terminal window, **from within the same directory as you run the nohup in**:

```
tail -f nohup.out
```

**IMPORTANT** these processes do not get killed when you terminate the session that executed the `nohup (...)` so you might need to do some manual `ps aux | grep ...` to find and kill them.

Source [here](https://linuxhint.com/how_to_use_nohup_linux/)

## Log in via ssh with username
when using ssh to log in as a specific user, do:
`ssh username@host_ip_address`
