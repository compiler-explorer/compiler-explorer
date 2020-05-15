caps.drop all
hostname ce-node
ipc-namespace
netfilter
private-dev
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

blacklist /boot
blacklist /sbin
blacklist /usr/local/sbin
blacklist /initrd*
blacklist /vmlinuz*
blacklist /usr/sbin
blacklist ${PATH}/su
blacklist ${PATH}/sudo
blacklist /lost+found
blacklist /media
blacklist /mnt
blacklist /root
blacklist /var
blacklist /snap
blacklist /srv

whitelist /opt/compiler-explorer
read-only /opt/compiler-explorer

read-only /celibs
read-only /infra

noexec /tmp
