#!/bin/bash
git rm --cached -r ./
git reset HEAD

# you can git rm any file you accidentity made
bash setupsshkey.sh
bash update_git.sh