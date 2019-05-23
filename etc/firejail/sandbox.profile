include etc/firejail/standard.inc

private-tmp
private-bin none
private-etc none
memory-deny-write-execute

# TODO: we would ideally allow ptrace to allow for address sanitizer/debuggers etc
# But can't find a way to blacklist everything in the default list *except* ptrace.
# Using seccomp.keep seems to turn things into a whitelist
#seccomp.keep ptrace

# Remove some env vars, mostly to stop people emailing me about them
# SUDO_COMMAND is one with actual somewhat sensitive info
rmenv SUDO_COMMAND
rmenv SUDO_USER
rmenv SUDO_UID
rmenv SUDO_GID
rmenv DBUS_SESSION_BUS_ADDRESS

# TODO are these appropriate values?
# rlimit-nproc seems not to be as useful as we want; it _seems_ to count *all* processes
# created by the effective user (i.e. would be shared across instances)
# rlimit-nproc 2000 # TODO test a fork bomb?
rlimit-fsize 16777216
rlimit-nofile 4
