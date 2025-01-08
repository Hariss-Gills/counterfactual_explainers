from tomllib import load


def main(file_path):
    with file_path.open("rb") as file:
        config = load(file)
    return config


if __name__ == "__main__":
    main("config.toml")
