import holopy as hp
imagepath = "difference.png"
raw_holo = hp.load_image(imagepath, spacing = 0.0851)
hp.show(raw_holo)