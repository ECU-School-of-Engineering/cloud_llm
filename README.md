# CLM Service Deployment Guide

This document describes how to run the CLM services locally, expose them externally, and configure tunneling between a local machine and EC2.

---

# V2.0 – clm_engine & llm_service

### Start CLM Engine

```bash
uvicorn clm_engine:app --host 0.0.0.0 --port 8080 --log-level info
```

### Start LLM Service

```bash
uvicorn llm_service:app --host 0.0.0.0 --port 8001 --log-level info
```

### Expose service using ngrok

```bash
ngrok http --url=swatheable-warier-kacey.ngrok-free.dev 8080
```

---

# Barry online


### Create reverse tunnel with autossh

```bash
AUTOSSH_DEBUG=1 autossh -M 0 -i /home/fhh/ivade.pem -p 9004 \
-o "ServerAliveInterval=30" \
-o "ServerAliveCountMax=3" \
-o "ExitOnForwardFailure=yes" \
-vvv \
-R 9000:localhost:8080 \
ec2-user@3.24.142.221
```

## Launch Additional Services – Barry (ecuonline)

### From Local PC
### Persistent autossh

```bash
autossh -M 0 -N \
-o "ServerAliveInterval=60" \
-o "ServerAliveCountMax=3"
```


#### Reverse SSH tunnel

```bash
ssh -i ~/.ssh/id_rsa -p 443 -f -N \
-R 0.0.0.0:8080:localhost:8080 \
ec2-user@54.253.1.8
```

#### Alternate SSH tunnel

```bash
ssh -i /home/fhh/ivade.pem -p 443 -vvv \
-R 8080:localhost:8080 \
ec2-user@3.24.142.221
```

#### AUTOSSH tunnel

```bash
AUTOSSH_DEBUG=1 autossh -M 0 -i /home/fhh/ivade.pem -p 9004 \
-o "ServerAliveInterval=30" \
-o "ServerAliveCountMax=3" \
-o "ExitOnForwardFailure=yes" \
-vvv \
-R 9000:localhost:8080 \
ec2-user@3.24.142.221
```

---

# EC2 Configuration

## Check Tunneling

```bash
ss -tlnp | grep 8080
```

Expected output:

```
LISTEN 0 128 0.0.0.0:8080 0.0.0.0:*
```

---

## Enable SSH Forwarding

Edit SSH configuration:

```bash
sudo nano /etc/ssh/sshd_config
```

Add or ensure the following settings exist:

```
GatewayPorts yes
AllowTcpForwarding yes
```

Restart SSH service:

```bash
sudo systemctl restart sshd
```

---

# Ports summary Barry Online with EC2

## Ports on Local PC

| Port | Purpose          |
| ---- | ---------------- |
| 8080 | ngrok            |
| 443  | ngrok outbound   |
| 9004 | AUTOSSH outbound |

---

## Ports on EC2

| Port | Purpose                  |
| ---- | ------------------------ |
| 9004 | SSH                      |
| 9000 | SSH reverse tunnel → CLM |
| 80   | HTTP                     |
| 443  | HTTPS                    |

---

# API Endpoints

## GET

```
admin/set_escalation/1?session_id=db0bdf7b-f7ef-40cd-be54-e375e42ccfaa
```

Full URL:

```
https://swatheable-warier-kacey.ngrok-free.dev/admin/set_escalation/1?session_id=db0bdf7b-f7ef-40cd-be54-e375e42ccfaa
```

---

## POST

```bash
curl -X POST "https://swatheable-warier-kacey.ngrok-free.dev/set_escalation?session_id=db0bdf7b-f7ef-40cd-be54-e375e42ccfaa&level=2.5"
```

---

# Notes

* `ngrok` is used to expose the local API externally.
* `autossh` maintains persistent reverse tunnels.
* EC2 must allow TCP forwarding and gateway ports for remote access.

## V1.0 – Legacy

### Start tmux session

```bash
tmux new -s clm_hume
tmux attach -t clm_hume
```

### Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```
