include etc/firejail/standard.inc

private-tmp
private-bin none
private-etc none
memory-deny-write-execute

nice 19

# Prevent sandbox talking to rsyslogd
blacklist /dev/log

# Prevent DoS on system-wide entropy generation
blacklist /dev/random

# No need to see anything here
blacklist /celibs
blacklist /compiler-explorer-image

# Remove some env vars, mostly to stop people emailing me about them
# SUDO_COMMAND is one with actual somewhat sensitive info
rmenv SUDO_COMMAND
rmenv SUDO_USER
rmenv SUDO_UID
rmenv SUDO_GID
rmenv DBUS_SESSION_BUS_ADDRESS

# TODO are these appropriate values?
rlimit-nproc 4
rlimit-fsize 16777216
rlimit-nofile 4
rlimit-as 536870912
