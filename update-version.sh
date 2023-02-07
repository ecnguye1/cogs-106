#!/bin/bash

git pull origin main

touch version.txt

date > version.txt

git add version.txt

git commit -m "New file added!"

git push --set-upstream origin main

git push
