---
tags: [PyCharm]
title: PyCharm
created: '2021-11-19T12:30:15.889Z'
modified: '2021-11-19T12:45:38.680Z'
---

# PyCharm

### PyCharm via ssh

<span style="color:red">**UNTESTED IDEA**</span>

If you want to edit a file on some linux server through PyCharm that is installed on your local machine (instead of using vim or nano) you will need to first mount the remote directory and adjust the project interpreter to refer to the remote python installation (or more specifically the virtual environment running on the remote machine).

#### Mount remote folder over ssh (Mac)
https://susanqq.github.io/jekyll/pixyll/2017/09/05/remotefiles/

#### Connect with PyCharm

<span style="color:red">**REQUIRES THE PROFESSIONAL LICENCE**</span>

- https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html 
- https://www.jetbrains.com/help/pycharm/accessing-files-on-remote-hosts.html
