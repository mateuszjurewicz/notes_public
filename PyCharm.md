# PyCharm

### General

- `cmd + alt + l`
reformats current active .py file according to Code Style settings. Mostly hard-wraps and alignments by default.
- `Zen Mode`
had to map this one manually via the preferences -> keymap, to `cmd + shift + 6`.
- `alt + tab`
toggle between opened split panes in pycharm. Needed in Zen mode.
- `ctrl + tab`
similar, hold ctrl and press tab and then navigate between open files seeing their names. Allows to go to other files opened in the same pane.

### PyCharm via ssh

<span style="color:red">**UNTESTED IDEA**</span>
<span style="color:red">**REQUIRES THE PROFESSIONAL LICENCE**</span>

- Seems like pycharm actually support editing files using SFTP (which uses SSH)
  - https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html#config
  - And remote Interpreters
https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html

- Video instruction requiring professional pycharm licence: https://www.youtube.com/watch?v=kuowBMqM1Ow&ab_channel=ArdianUmam

If you want to edit a file on some linux server through PyCharm that is installed on your local machine (instead of using vim or nano) you will need to first mount the remote directory and adjust the project interpreter to refer to the remote python installation (or more specifically the virtual environment running on the remote machine).

#### Mount remote folder over ssh (Mac)
https://susanqq.github.io/jekyll/pixyll/2017/09/05/remotefiles/

#### Connect with PyCharm

- https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html 
- https://www.jetbrains.com/help/pycharm/accessing-files-on-remote-hosts.html
