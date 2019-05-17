caps.drop all
hostname ce-node
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

# blacklisting this directory to ensure the "cwd" of the process is not valid when executed.
# this forces firejail to chdir("~") before running, which we rely on. Also good hygiene in
# general, even though the image is pretty much open source.
blacklist /compiler-explorer-image

blacklist /sbin
blacklist /usr/local/sbin
blacklist /initrd*
blacklist /vmlinuz*
blacklist /usr/sbin
blacklist ${PATH}/su
blacklist ${PATH}/sudo
blacklist /lost+found
blacklist /var
blacklist /snap
blacklist /srv

whitelist /opt/compiler-explorer
read-only /opt/compiler-explorer

noexec /tmp
