@echo off
setlocal enabledelayedexpansion

REM Usage: get_last_version_of_lib.bat package_name
if "%~1"=="" (
    echo Usage: get_last_version_of_lib.bat package_name
    exit /b 1
)
set "pkg_name=%~1"
set "last_version="

REM Loop over files in the dist folder matching "pkg_name-*.tar.gz"
for /f "delims=" %%F in ('dir /b "dist\%pkg_name%-%*.tar.gz" 2^>nul') do (
    set "fname=%%F"    
    echo found version of !pkg_name!: !fname! : !ver!
    REM Example: common_tools-0.4.11.tar.gz
    REM Remove the package name and dash to get "0.4.11.tar.gz"
    set "ver=!fname:%pkg_name%-=!"
    REM Remove the .tar.gz suffix (assumes no extra dot in the version)
    for %%A in ("!ver!") do set "ver=%%~nA"
    if "!last_version!"=="" (
        set "last_version=!ver!"
    ) else (
        REM Use lexicographical comparison (assuming consistent format)
        if "!ver!" gtr "!last_version!" (
            set "last_version=!ver!"
        )
    )
)

if defined last_version (
    echo !last_version!
) else (
    echo 0
)
endlocal
