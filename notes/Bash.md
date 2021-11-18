---
favorited: true
tags: [Notebooks/Unix, Unix]
title: Bash
created: '2021-11-18T10:43:28.058Z'
modified: '2021-11-18T12:11:24.390Z'
---

# Bash
Based on the video: https://www.youtube.com/watch?v=oxuRxtrO2Ag

### Basic
* `~` sign always refers to the home directory.				
* `/` (root) sign refers to where every other directory is on a linux / unix machine.	
* `ls` tells you what is in your current directory							
* `ls -a` shows hidden (all) documents and directories							
* `ls -l` shows a long list of information about non-hidden contents of pwd, e.g. size, date, permission							
* `ls -lh` shows more info in human readable format (combined flags)							
* `pwd` tells you what your present working directory / folder is (same thing on unix).							

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
* `locate sth` uses a database and shows every file / folder with sth in its name, from pwd. Sometimes requires creation of this database first (updated once a day)						
* `sudo updatedb` manually start an update to the locate database (requires root privileges)
* `find . -name *sth*` looks for anything that fits the pattern *sth* in the . directory.						
* `which sth` tells you if and where sth (some command, like ls) is installed on the system
* `tree` gives you a easy-to-read tree of things in pwd							
								
### Commands info						
* _up arrow_	goes up in the order of execution of previous commands
* `history`	prints a list of 1 thousand executed commands							
* `man some_command` gives full manual info about a command, including flags requires pressing :q to exit the manual (I think it opened vim)
* `whatis some_command` gives a short manual about a command breaks a bit if multiple similar commands are available							
* `apropos` sth	gives you a list of commands with sth in their names, so that you can look them up

### Editing files and directories

* `mkdir a_dir` makes a directory. You can type multiple names to create many in one go.	
* `touch a_file` either creates a file if it doesn't exist or updates its last modified date if it does.							
* `cp from/file1 to/file2` copies file1 from first directory onto the second one, under name file2 using TAB to make sure the original file exists is good practice.						
* `mv file1 file2` moves / renames a file							
* `rm sth` removes a file							
* `rm -r dir1` removes recursively the directory and its contents							
* `rmdir dir1 dir2` removes only empy directories from given list	you can use it with * to clean things safely (only empty directories) matching some name pattern
* `head file1` shows first 10 lines of a file.
* `head -20 file1` shows first 20 lines.
* `cat file1` prints the entire contents of a file							
* `cat >> file2` opens a console for writing things in, then pressing ctrl+c or ctrl+d will close it end we'll write-append that into file2 (possibly creating it)							
* `cat > file3` opens a console for writing things in, then overwrites (instead of appending) the content of file3 with what we wrote.

