# Running a remote compiler on Azure

Instructions to get CE to run on an Azure vm running Visual Studio 2017 Community.

The VM used in this example is the "Visual Studio Community 2017 (latest release) on Windows Server 2016 (x64)" by Microsoft (2018-01-03)

Instructions to run CE on the Windows machine:
* Log onto RDP
* Install NodeJS (v8)
* Download or clone CE
* Create the folder c:\tmp\
* From a command line
  - Run 'npm remove newrelic'
  - Run 'npm install'
  - Run 'azure-app.cmd'
* Allow TCP port 10240 through the Windows firewall
* Allow TCP port 10240 through the Azure network settings

Automatic startup on the Windows machine:
* Create a new Task with the Task Scheduler
* Set the startup Trigger to 'At startup', disable 'stop task' options
* Set the action to starting azure-app.cmd in your CE folder
* Set the initial working directory to your CE folder
* Set the security option to 'Run whether user is logged on or not'

From your *nix CE installation:
* Add your Azure machine as a remote by adding it to your C++ compilers properties file
  - In the format 'azurevm-ipaddress@10240'
