from argparse import ArgumentParser
import yaml


def extract_port(addr: str) -> str:
    return addr.rsplit(":", 1)[-1]


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Generate .env from network config")
    arg_parser.add_argument(
        "--network_config",
        type=str,
        default="configs/network.yaml",
        help="Path to network configuration YAML file",
    )
    args = arg_parser.parse_args()

    with open(args.network_config) as f:
        cfg = yaml.safe_load(f)

    env = {}

    env["HUB_PORT"] = extract_port(cfg["zmq"]["hub_endpoint"])
    env["PERCEPTION_PORT"] = extract_port(cfg["zmq"]["perception_endpoint"])
    env["control_PORT"] = extract_port(cfg["rest"]["control_endpoint"])

    with open(".env", "w") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")

    print("Generated .env:")
    for k, v in env.items():
        print(f"{k}={v}")
