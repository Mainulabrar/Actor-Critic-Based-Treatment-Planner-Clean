#!/bin/bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/SSH_personal/id_rsa
ssh-add -l -E sha256
