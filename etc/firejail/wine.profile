include etc/firejail/standard.inc

private-etc passwd,fonts

whitelist /opt/wine-stable
read-only /opt/wine-stable
