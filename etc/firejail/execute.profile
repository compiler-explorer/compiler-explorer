include etc/firejail/standard.inc

private-tmp
private-etc passwd
ipc-namespace

whitelist /opt/intel
read-only /opt/intel
