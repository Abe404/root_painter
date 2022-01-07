import shutil


def clean():
    """
    Deletes the build target directory.
    """
    try:
        shutil.rmtree("./target")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    clean()
