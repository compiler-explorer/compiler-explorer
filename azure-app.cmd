@echo off
./node_modules/.bin/supervisor.cmd -s -e node,js,properties -w app.js,etc,lib -- app.js --env azure --port 10240 --tmpDir=c:/tmp/
