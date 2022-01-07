from settings import Settings
from clean import clean
from freeze import freeze
from installer import create_installer


def main(settings=Settings()):
    clean()
    freeze(settings)
    create_installer(settings)


if __name__ == "__main__":
    main()
