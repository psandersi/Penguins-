"""Download the original simplified dataset once; later runs use the local file."""
from urllib.request import urlopen

from penguins.config import RAW

URL = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv'
DOWNLOAD_TIMEOUT = 60


def main():
    if RAW.exists():
        return

    RAW.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=DOWNLOAD_TIMEOUT) as response:
        content = response.read()

    # Do not leave a partial CSV for the next run to reuse.
    temporary = RAW.with_suffix('.tmp')
    temporary.write_bytes(content)
    temporary.replace(RAW)


if __name__ == '__main__':
    main()
