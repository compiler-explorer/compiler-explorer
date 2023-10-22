# Using Systemd socket based activation to start Compiler Explorer

This document gives a short overview of how to use Systemd to automatically start Compiler Explorer when the
web-interface is accessed.

You'll need to create two files in `/etc/systemd/system/`:

compiler-explorer.socket:

```
[Socket]
ListenStream=10240

[Install]
WantedBy=sockets.target
```

compiler-explorer.service:

```
[Service]
Type=simple
WorkingDirectory={{path_to_installation_directory}}/compiler-explorer
ExecStart=/usr/bin/node {{path_to_installation_directory}}/compiler-explorer/out/dist/app.js
TimeoutStartSec=60
TimeoutStopSec=60
User={{run_as_this_user}}
Group={{run_as_this_group}}
```

Replace the bracketed `{{}}` placeholders with your system specifics.

Once the two above files are created Systemd needs to be made aware of the changes:

```sh
sudo systemctl daemon-reload
```

Now all that remains is to enable and start the new service:

```sh
sudo systemctl enable compiler-explorer.socket
sudo systemctl start compiler-explorer.socket
```

If all goes well you can now open the web-interface.
