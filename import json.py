import json

# Suppose config.json looks like this:
# {
#     "use_cpu": true,
#     "use_gpu": false
# }

with open("config.json", "r", encoding="utf-8") as f:
    print(f.read())
    config = json.load(f.read())

# print(config)
# # -> {'use_cpu': True, 'use_gpu': False}

# print(config["backend"])  # True