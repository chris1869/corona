import sys
import re
import os
import boto3

config = {}

s = boto3.Session()
for skey in ["aws_access_key_id", "aws_secret_access_key"]:
    config[skey.upper()] = s._session._config["profiles"]["default"][skey]

pattern = "^([\s\w\":\-}{/,]*)\\${(\w*)}(.*)$"
env_re = re.compile(pattern)

finput = sys.argv[1]
foutput = sys.argv[2]

fo = open(foutput, "w")
#fo.write("\uFEFF")

with open(finput, "r") as fi:
    for line in fi:
        while True:
            match = env_re.match(line)
            if match is None:
                break
            else:
                try:
                    line = match.group(1) + os.getenv(match.group(2)) + match.group(3) + "\n"
                except:
                    line = match.group(1) + config[match.group(2)] + match.group(3) + "\n"
        fo.write(line)
fo.close()

import json

c = boto3.client("ecs")

with open(foutput, "r") as fjson:
    params = json.load(fjson)
    c.register_task_definition(**params)
