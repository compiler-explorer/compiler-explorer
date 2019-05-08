include disable-common.inc
include disable-programs.inc

caps.drop all
hostname ce-node
ipc-namespace
netfilter
private-dev
private-tmp
net none
no3d
nodbus
nodvd
nogroups
nonewprivs
noroot
nosound
notv
nou2f
novideo
seccomp
x11 none

shell none
disable-mnt

blacklist /lost+found
blacklist /var
blacklist /snap
blacklist /srv
whitelist /opt/compiler-explorer
read-only /opt/compiler-explorer

noexec /tmp
