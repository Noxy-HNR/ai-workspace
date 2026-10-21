# Loading models from the home server

The dashboard, `start.ps1` and Model Tuner can load GGUF files straight from the HP model server's
Samba share. llama-server on this PC reads the file over SMB once (`--load-mode none`) and runs
inference locally. Nothing is copied to this PC, and there is no local cache: the server's five-disk
RAID 0 array reads faster than the network can carry, so the network link is the limit either way.

A few models can be **pinned** to this PC. They keep loading from `C:\AI\models` when the server
also has them, and they still work when the server is off or the laptop is away from home.

## Configure the server path

Copy `Config\model-storage.example.json` to `Config\model-storage.json` and edit it:

```json
{
  "networkDir": "\\\\192.168.1.241\\ModelFiles\\Models",
  "pinnedLocal": [
    "Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf"
  ]
}
```

| Key | Meaning |
|---|---|
| `networkDir` | The folder on the server that holds the `.gguf` files, as a UNC path. Use the UNC path, not a mapped drive letter such as `M:`: drive letters belong to one logon session, and the dashboard can run in another one. LoRA adapters go in its `lora` subfolder. |
| `pinnedLocal` | File names that load from `C:\AI\models` even when the server has a copy. |
| `networkHost` | Optional. The host to probe; it defaults to the host in `networkDir`. |

`Config\model-storage.json` is ignored by Git, since the path is specific to each PC. Without the
file, or without `networkDir`, everything loads from `C:\AI\models` exactly as before.

### Which copy loads

| Model | Server reachable | Server unreachable |
|---|---|---|
| Not pinned, on the server | Server copy | Local copy if there is one, else a clear error |
| Pinned | Local copy (server copy if there is no local one) | Local copy |
| Only on this PC | Local copy | Local copy |

New files on the share appear in the Models tab on its next refresh. A file the server changed in
the last minute is skipped until it settles, so a model that is still being copied never shows up
half-written. Each model card says where it loads from ("Model server", "This PC" or "This PC
(pinned)"). A bare `--mmproj` file name in a model's extra arguments is looked up next to the model
first, so a server model takes its projector from the server.

### Credentials

The share is restricted to the server's `lab` account. Save the credentials for the exact host
name or IP used in `networkDir`, under the Windows account that runs the dashboard:

```powershell
cmdkey /add:192.168.1.241 /user:lab /pass
Test-Path '\\192.168.1.241\ModelFiles\Models'   # should print True
```

## When the share can't be reached

Before touching the share, the dashboard checks TCP port 445 on the server with a one-second limit
(at most once every 15 seconds), so an unplugged server can't stall the dashboard. If the server is
unreachable, the Models tab shows a yellow notice with the reason and lists this PC's models only.
The messages and what they mean:

| Message starts with | Likely cause | What to check |
|---|---|---|
| "did not answer on the SMB port (TCP 445) within 1 second" | Server off, wrong IP, or this PC is on another network | `Test-NetConnection 192.168.1.241 -Port 445` |
| "refused the SMB connection" | Server is up but Samba is not running | On the server: `systemctl status smbd` |
| "is up, but ... can't be opened" | Wrong share or folder name, or no saved credentials | Open the path in Explorer; run the `cmdkey` line above |
| "name ... could not be resolved" | `networkDir` uses a host name that DNS doesn't know | Use the IP address |
| "networkDir must be a UNC path" | `networkDir` is a drive letter or a plain folder | Use `\\host\share\folder` |

Loading a model that is only on the server while the server is down fails before the running model
is stopped, so the current model keeps serving. If the share drops while a model is loading,
llama-server exits; the dashboard reports that within a few seconds, with the last error from
`Logs\llama.stderr.log` and the server's state, instead of waiting out the 10-minute load timeout.

## SMB multichannel

SMB 3 multichannel lets one SMB session use several network paths at once. The HP has four
Broadcom BCM5719 1 GbE ports. With each port cabled and given its own IP address (no LACP bond),
Windows opens a connection to each server address and spreads a large read across them.

> **Current state:** the server inventory from 2026-10-20 (`homelab-hp`, `docs/HP-CURRENT-STATE.md`)
> shows only `eno1` cabled, at `192.168.1.241`. Until the other ports are cabled and addressed,
> transfers use that single 1 GbE path. The model loading code doesn't depend on the number of
> links; the setup below is only about speed.

### Server (Ubuntu, Samba)

1. Give each cabled port its own address on the LAN, for example `.241` to `.244`. Because they share
   one subnet, Linux needs two things so each address answers on its own port: strict ARP, and a
   routing table per port so replies leave by the port they arrived on. With netplan:

   ```yaml
   # /etc/netplan/60-smb-multichannel.yaml  (one block per cabled port; eno1 and eno2 shown,
   # eno3/eno4 follow the same pattern with tables 103/104). Replace the router address if it differs.
   network:
     version: 2
     ethernets:
       eno1:
         addresses: [192.168.1.241/24]
         routes:
           - { to: default, via: 192.168.1.1 }                       # main table: only on eno1
           - { to: 192.168.1.0/24, scope: link, table: 101 }
           - { to: default, via: 192.168.1.1, table: 101 }
         routing-policy:
           - { from: 192.168.1.241, table: 101 }
       eno2:
         addresses: [192.168.1.242/24]
         routes:
           - { to: 192.168.1.0/24, scope: link, table: 102 }
           - { to: default, via: 192.168.1.1, table: 102 }
         routing-policy:
           - { from: 192.168.1.242, table: 102 }
   ```

   ```bash
   # /etc/sysctl.d/60-smb-multichannel.conf
   net.ipv4.conf.all.arp_ignore = 1
   net.ipv4.conf.all.arp_announce = 2
   ```

   Apply with `sudo netplan try` (it rolls back if you lose the connection) and
   `sudo sysctl --system`. Reserve the addresses in the router's DHCP so nothing else takes them.

2. In `/etc/samba/smb.conf`, under `[global]`:

   ```ini
   server multi channel support = yes
   # Optional: only needed if Samba can't read the link speed itself.
   interfaces = "192.168.1.241;speed=1000000000" "192.168.1.242;speed=1000000000" "192.168.1.243;speed=1000000000" "192.168.1.244;speed=1000000000"
   ```

   Multichannel has been supported (not experimental) since Samba 4.15, and Ubuntu 26.04 ships a
   newer release. Check with `testparm -s`, then `sudo systemctl restart smbd` when no transfer
   is running.

Keep `networkDir` on one address (normally `.241`). Windows connects to that address, then learns
the others from the server and adds connections to them. If that one port is down, the first
connection fails, so point `networkDir` at the most reliable link.

### Windows client

Multichannel is on by default. To confirm, and to see it working:

```powershell
(Get-SmbClientConfiguration).EnableMultiChannel        # True
# Start a model load from the server, then while it reads:
Get-SmbMultichannelConnection | Format-Table ServerIpAddress, ClientIpAddress, ClientLinkSpeed
Get-SmbConnection | Format-Table ServerName, ShareName, Dialect   # Dialect 3.1.1
```

Expect one row per server address in `Get-SmbMultichannelConnection`. A single row means the
server isn't advertising its other ports (check step 1 and `smbstatus`), or only one is cabled.

### What to expect

| Laptop link | Server links in use | Rough ceiling | 15 GB model |
|---|---|---|---|
| Wi-Fi (about 540 Mb/s) | any | 50-60 MB/s | about 5 min |
| 1 GbE | 1 or more | about 115 MB/s | about 2 min 10 s |
| 2.5 GbE | 1 | about 115 MB/s | about 2 min 10 s |
| 2.5 GbE | 3 or 4 | about 280 MB/s | about 55 s |

The laptop's own link is the cap: a 2.5 GbE port can use about two and a half of the server's
1 GbE links, so cabling a third and fourth port mainly helps when two PCs read at once. These
figures are line-rate estimates, not measurements on this hardware. Measure a real load by
watching the dashboard's load timer, or with `Measure-Command { Copy-Item ... }` to a scratch folder.
