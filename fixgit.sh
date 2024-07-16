#!/bin/bash
git rm --cached -r ./
git reset
bash setupsshkey.sh
bash update_git.sh