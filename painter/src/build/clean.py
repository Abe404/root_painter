import shutil


def clean():
    """
    Deletes the build target directory.
    """
    shutil.rmtree("./target", ignore_errors=True)


if __name__ == "__main__":
    clean()
