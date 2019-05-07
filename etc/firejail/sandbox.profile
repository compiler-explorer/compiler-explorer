include disable-common.inc
include disable-programs.inc

caps.drop all
hostname ce-node
ipc-namespace
netfilter
private-dev
private-tmp
private-bin none
private-etc none
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

# TODO: we would ideally allow ptrace to allow for address sanitizer/debuggers etc
# But can't find a way to blacklist everything in the default list *except* ptrace.
# Using seccomp.keep seems to turn things into a whitelist
#seccomp.keep ptrace
shell none
disable-mnt
memory-deny-write-execute

noexec /tmp

blacklist /lost+found
blacklist /var
blacklist /snap
blacklist /srv
whitelist /opt/compiler-explorer
read-only /opt/compiler-explorer

# TODO need to launder the environment more before executing

# TODO are these appropriate values?
# rlimit-nproc seems not to be as useful as we want; it _seems_ to count *all* processes
# created by the effective user (i.e. would be shared across instances)
# rlimit-nproc 2000 # TODO test a fork bomb?
rlimit-fsize 16777216
rlimit-nofile 4
