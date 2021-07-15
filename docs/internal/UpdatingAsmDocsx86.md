If you need to update the x86 asm documentation, just run `etc/scripts/docenizer.py`, which requires:
 - Python 3.x with BeautifulSoup

You can use some options in the script:
 - `-o`/`--outputpath` - Final destination of the generated JavaScript file
 - `-i`/`--inputfolder` - Points to the downloaded and extracted .html files
 - `-d`/`--downloadfolder` - Points to the download folder to use in case a new version is needed

Now you only need to run it, and hope for the best. Please report any problems you encounter
