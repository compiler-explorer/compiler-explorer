@echo off
set INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.12.25827\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.16299.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.16299.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.16299.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.16299.0\winrt
set LIB=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.12.25827\lib\x86
set LIBPATH=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.12.25827\lib\x86
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x86\
cl %*
