import yaml

with open("../configs/basic_yaml.yaml", "r") as f:
    basic_config = yaml.safe_load(f)

for config in basic_config:
    print(config)
    print(basic_config[config])
    print("="*50)
