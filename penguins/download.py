"""Download the original simplified dataset once; later runs use the local file."""
from urllib.request import urlopen
from .config import RAW

URL = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv'

def main():
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    temporary = RAW.with_suffix('.tmp')
    temporary.write_bytes(content)
    temporary.replace(RAW)

if __name__ == '__main__':
    main()
