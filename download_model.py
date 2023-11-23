import gdown
output = "./model.rar"
url = "https://drive.google.com/uc?export=download&id=1rlZcBfkeQO1E9sWNJ9jpsJy66QC7l_3P"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)