---
tags: [Notebooks/Skills/Tech]
title: Bash
created: '2021-11-18T10:43:28.058Z'
modified: '2021-12-10T14:06:50.803Z'
---

# Bash
Based on the video: https://www.youtube.com/watch?v=oxuRxtrO2Ag

### Shortcuts
* `ctrl + d` = `exit`
* `ctrl + c` kills current process
* `ctrl + l` = `clear`

### Basic

* `~` sign always refers to the home directory.				
* `/` (root) sign refers to where every other directory is on a linux / unix machine.	
* `ls` tells you what is in your current directory							
* `ls -a` shows hidden (all) documents and directories							
* `ls -l` shows a long list of information about non-hidden contents of pwd, e.g. size, date, permission					
* `ls -lh` shows more info in human readable format (combined flags)
* `pwd` tells you what your present working directory / folder is (same thing on unix).	
* `ls -alh /some_dir` will print the contents of a specified directory, with all 3 flags.
* `ls -Sh` will list sorting by size (from largest) (`-S`), human readable.

### Navigating directories

- `cd` used for moving around, without any arguments takes you to your home (~) directory.		
- `.` current directory							
- `..` one-up directory							
- `TAB` used for auto-complete, press twice to see candidates							
- `\` back-slash is used for escaping characters, particularly spaces in directory and file names		
- `/` forward-slash is also used for addressing, e.g. ~/usr/mateusz	(other than just meaning root dir)									
- `pushd /some_dir` takes you to /sdir, but remembers where you started the journey							
- `popd /some_dir` takes you back to your original directory, before moving to /sdir							
- `file some_file` gives you info about a dir or file (e.g. is it jpeg, txt, wav etc., because on unix we don't always have extensions)							
								
### Finding things				

* `locate sth` uses a database and shows every file / folder with sth in its name, **from pwd**. Sometimes requires creation of this database first (updated once a day)						
* `sudo updatedb` manually start an update to the locate database (requires root privileges)
* `find . -iname *sth*` looks for anything that fits the pattern *sth* in the . directory. The `-name` or `-iname` flag specifies to look for the string pattern in the name of the file in a case sensitive or insensitive way. Otherwise it will look for the pattern also in the contents of files.			
* `which sth` tells you if and where sth (some command, like ls) is installed on the system
* `tree` gives you a easy-to-read tree of things in pwd							
								
### Commands info						

* `up arrow`	goes up in the order of execution of previous commands
* `history`	prints a list of 1 thousand executed commands							
* `man some_command` gives full manual info about a command, including flags requires pressing :q to exit the manual (I think it opened vim)
* `whatis some_command` gives a short manual about a command, breaks a bit if multiple similar commands are available							
* `apropos` sth	gives you a list of commands with sth in their names, so that you can look them up

### Moving, deleting and renaming files and directories

* `mkdir a_dir` makes a directory. You can type multiple names to create many in one go.	
* `touch a_file` either creates a file if it doesn't exist or updates its last modified date if it does.					
* `cp from/file1 to/file2` copies file1 from first directory onto the second one, under name file2 using TAB to make sure the original file exists is good practice.						
* `mv file1 file2` moves / renames a file							
* `rm sth` removes a file							
* `rm -r dir1` removes recursively the directory and its contents							
* `rmdir dir1 dir2` removes only empy directories from given list	you can use it with * to clean things safely (only empty directories) matching some name pattern

### Reading and editing files

* `head file1` shows first 10 lines of a file.
* `head -20 file1` shows first 20 lines.
* `tail -20 file1` shows last 20 lines.
* `cat file1` prints the entire contents of a file.
* `cat >> file2` opens a console for writing things in, then pressing ctrl+c or ctrl+d will close it end we'll write-append that into file2 (possibly creating it).
* `cat > file3` opens a console for writing things in, then overwrites (instead of appending) the content of file3 with what we wrote.
* `cat file1 file2` will print out the concatenated contents of the 2 files, in order.
* `more file1` gives a way to quickly read the entire file quickly, using space to move from screen-worth to another. Use `q` to exit.
* `less file1` a better version of more, allows for arrow key usage and searching.

### Nano

**nano** is an alternative to `vim`, with nano you're immediately in edit mode and the magic key is `ctrl`, which is what you press with other keys to save (`ctrl + o`) and exit (`ctrl + x`). You can also search (ask "where is?" via `ctrl + w`).

You can also create a new file directly by executing `nano file1`. 

More about Nano in [[Nano]]

### Redirection

* `>` will overwrite from left to right
* `>>` will append
* `|` will direct output of one command as input to another.
E.g. `history | less` will let us scroll through the command history using the spacebar (via `less`)

### Users

* `sudo -s` will (for a short time) make all our commands be executed as the superused (avoid).
* `su - some_user` will switch current user to some_user and move us to their home directory.
* `su some_user` as above, without changing directories.
* `exit` will log you out as that user.
You can also use `ctrl + d`, does the same thing! 
* `su` by itself changes the user to root (without moving to their home dir, for that `su -`)
* `users` prints active users on the system (e.g. multiple via separate terminals, via ssh)
* `id` gives you your current user id

### Permissions

The permissions for files are reported in 3 sectons of 3 symbols (9 total).
First section (leftmost) is current user, second section is current user's group, third is everyone.
E.g a file might have the following permissions:
```-rwxr-xr-x  13 mjure  staff   416 Nov 18 15:57 .```
These stand for `r` reading. `w` writing, `x` executing.

* `chmod +x file1` - changes permissions so that everyone can do anything to file1
* `chmod -x file1` - removes the `x` excute permission from everyone for file1.
* `chmod 700 file1` - gives me (first of 3 digits) full permissions (7) and nothing to the other 2 section (00)

`chmod 4` will give read access, `chmod 6` will give read and write access. They go down in increments of `421`.

Commonly you set `chmod 644` for files (everyone can read, I can read + write) and `chmod 755 /a_dir` for directories. Changing permissions on a directory can make it invisible to some users.

### Processes

You want to sometimes monitor or kill processes.
* `watch free -h` executes the `free` command with human-readable, so every 2 sec we see memory usage.
* `ctrl + c` kills the current command
* `killall app_name` kills the application / processes with given name

### Bash Script

How to write shell scripts for bash.

- `!#/bin/bash`
Has to be put up top in the script, it tells the system what program to use to execute this script (and that it is a script to begin with, via the hash-bang aka shebang: `!#`).

- `.sh` extension
is not strictly necessary, but good practice when naming bash script files. `sh` refers to a wider, more universal verson of the scripting language that bash is built on top of.
- `chmod 700 first.sh`
you may need to change the permission to make the file executable first, which will often show it in a different color in the console. However, I was able to run the file prior to this via `bash first.sh`. This however, enables running it via `./first.sh`.
- `!#/bin/bash -x`
this will run your script in debug mode, with more print-outs as to what's going on.
- `~/username/bin` or `/usr/local/bin` is where you would usually put your .sh scripts so that you don't have to point to them via relative positioning (`./first.sh`), and then just use `first.sh` - but it doesn't quite work on a Mac, cause presumably that dir is not in the path variable.
- `echo $PATH` or better `echo $PATH | tr : '\n'`
will print out exactly what's in the current path, which is where the system will look for executable commands (as .sh scripts). This is often set / adjusted based on the contents of `.bashrc` or `.bash_profile` (on Mac, possibly just `.profile` on a linux). 
- `sudo first.sh` will only work if your script is in `usr/local/bin`, not in e,g, `/Users/mj/bin`. Otherwise you can't execute it via sudo. That's because sudo is like using the root user to execute sth, and the root user will not look for executables in your personl home directory.
- `\` backslash allows to break long lines into multiple lines.
- `/dev/null` 
is the trash bucket folder, sometimes you'll want to only know if a command executed (with what status), not see its output, so then you can do `cmd.sh > /dev/null` and the actual output that would've gone to `stdout` (your screen) will be deleted. 
- `&1` and `&2` can be used to redirect output to `stdout` and `stderr` respectively.
E.g. `echo 'This will go to stdout' >&1` will got to stdout. 
**Note the lack of space after > redirection!**
- `env` 
prints environment variables in the current shell.
- `a=b` sets a variable
  - **gotcha** there has to be no spaces around the equals sign :)
  - `echo $a` prints a variable (note the `$`)
  - `unset a` to remove a variable (note the lack of $)
- `$@` represents all of the arguments passed to a command in numerical order (can be used in a loop)
  - and `$0` is the name of the command, `$1` is the first argument and so forth.
- `test` 
is a condition evaluation utility with many flags
  - `test $a == "b"` will give a 0 exit code, and can be used to test booleans
  - you can check whether a file is present using the `test -f` command with a flag:
  `test -f ".bashrc" && echo "bashrc is here" || "bashrc is not here"`
- `&&` is used for sequentially chaining commands.
  - `test $a == "b && echo "Worked"` shows how to run commands sequentially via the `&&` double ampersand, giving us a nice printout if the test is successful (otherwise we have no way of knowing). 
- `||` can be used for if-not, in conjunction with the &&
  - `test $a == "b" && echo "a equals b" || echo "a doesn't equal b"`
- `!=` and `==` are not-equal and equals checks
- `$?`
is the variable storing the exit code of the lasft command. 0 is good, everything else is not.

- **if / else** conditionals:
```
#!/bin/bash

a=b
if [ $a == "b" ]; then
        echo "a equals b"
else
        echo "a doesn't equal b"
fi
```
- **send output you don't care about to** `/dev/null`
```
#!/bin/bash

which ifconfig >/dev/null
if [ $? != "0" ]; then
        echo "SyncAll Error: the ifconfig is not installed!" >&2
        echo "Ubuntu and Linux Mint: Run 'sudo apt install net-tools'" >&2
        exit 1
else
        echo "ifconfig is installed!"
fi
```


