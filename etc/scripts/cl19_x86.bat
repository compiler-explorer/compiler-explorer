@echo off
rem gotten from something like:
rem call "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
rem set
rem get out the include/lib/libpath/path
set INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\INCLUDE;C:\Program Files (x86)\Windows Kits\10\include\10.0.10240.0\ucrt;C:\Program Files (x86)\Windows Kits\8.1\include\shared;C:\Program Files (x86)\Windows Kits\8.1\include\um;C:\Program Files (x86)\Windows Kits\8.1\include\winrt;
set LIB=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB;C:\Program Files (x86)\Windows Kits\10\lib\10.0.10240.0\ucrt\x86;C:\Program Files (x86)\Windows Kits\8.1\lib\winv6.3\um\x86;
set LIBPATH=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB;References\CommonConfiguration\Neutral;\Microsoft.VCLibs\14.0\References\CommonConfiguration\neutral;
set Path=C:\Program Files (x86)\MSBuild\14.0\bin;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\BIN;C:\Program Files (x86)\Windows Kits\8.1\bin\x86;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files\Amazon\cfn-bootstrap\;C:\ProgramData\chocolatey\bin;C:\Program Files\nodejs\;C:\Program Files\Git\cmd;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Users\Administrator\AppData\Roaming\npm
cl /nologo %*
