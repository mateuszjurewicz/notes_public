# Git & Github


## General
- `cmd + k` allows you to quickly search within-repository for terms in the browser.

## Change Remote Origin
Sometimes you may have cloned someone's repository without making a fork of it (a proper repository of your own, but a copy of the original one, in a way that's linked and visible on Github). If you want to make any local changes you've made tracked in a properly linked & forked repo, you need to:

1. Fork the target repository via the GitHub website
2. Grab the ssh of your forked repo, e.g. 

`git@github.com:user/forked_repo.git`

3. Set your local repo's remote origin to the above, via

 `git remote set-url origin git@github.com:user/forked_repo.git`
 
4. Push your local changes to your forked repo.