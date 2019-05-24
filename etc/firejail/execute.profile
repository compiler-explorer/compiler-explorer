include etc/firejail/standard.inc

private-tmp
private-etc passwd

# Prevent modification of anything left over from the rootfs
read-only /

nice 10
rlimit-as 536870912
whitelist /opt/intel
read-only /opt/intel
