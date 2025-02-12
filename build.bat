@echo off
echo Cleaning previous build...
rmdir /s /q build
rmdir /s /q release

echo Creating release directory...
mkdir release

echo Installing dependencies...
pip install pyinstaller

echo Building executable...
pyinstaller --onefile --name JunoModel ^
--add-data ".env;." ^
--add-data "scripts;scripts" ^
--add-data "utils;utils" ^
--add-data "predictions;predictions" ^
main.py

echo Build complete! Executable is in the release/ folder.
pause
