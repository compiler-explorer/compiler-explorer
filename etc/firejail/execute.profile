include etc/firejail/standard.inc

private-tmp
private-etc passwd,ld.so.conf.d,ld.so.conf

# Prevent modification of anything left over from the rootfs
read-only /

nice 10
# 1.25GB should make two compiles fit on our ~3.8GB machines
rlimit-as 1342177280
whitelist /opt/intel
read-only /opt/intel
whitelist /opt/arm
read-only /opt/arm
