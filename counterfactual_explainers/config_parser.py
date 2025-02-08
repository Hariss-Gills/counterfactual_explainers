from tomllib import load


# TODO: This script should simply upload a user's local config to packages path
def main(file_path):
    with file_path.open("rb") as file:
        config = load(file)
    return config


if __name__ == "__main__":
    main("config.toml")
