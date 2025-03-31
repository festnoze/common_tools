

@echo off
REM Remove previous build artifacts
REM rmdir /s /q common_tools\__pycache__
rmdir /s /q common_tools.egg-info

REM Define an env. for the local variables created in this script
setlocal

REM Calculate the version of the lib to be created based on the previous existing versions
call get_next_version_of_lib.bat common_tools

REM NEW_VERSION is now set in the parent shell.
echo Next version to build is: %NEW_VERSION%
set "BUILD_VERSION=%NEW_VERSION%"
echo The version of lib that will be built is: %BUILD_VERSION%

REM Run the build command (setup.py reads BUILD_VERSION from the environment)
python -m build .
echo package: common_tools-%BUILD_VERSION% is now available into the dist folder
endlocal