import json

# Suppose config.json looks like this:
# {
#     "use_cpu": true,
#     "use_gpu": false
# }

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print(config)
# -> {'use_cpu': True, 'use_gpu': False}

print(config["use_cpu"])  # True
print(config["use_gpu"])  # False
