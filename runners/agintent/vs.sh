#!/bin/zsh
# ssh to an agintent Vast box: zsh vs.sh TAG 'command'
D=<repo root>/.runtime; T=$1; shift
read H P < <(vastai show instance $(cat $D/vast_instance_id_ai_$T) --raw 2>/dev/null | python3 -c "
import json,sys;d=json.load(sys.stdin);p=(d.get('ports') or {}).get('22/tcp')
print(d['public_ipaddr'].strip(), p[0]['HostPort']) if p else print(d['ssh_host'], d['ssh_port'])")
exec ssh -i $HOME/.ssh/vast_agent -o StrictHostKeyChecking=no -o UserKnownHostsFile=$D/known_hosts_vast -o ConnectTimeout=15 -o LogLevel=ERROR -p $P root@$H "$@"
