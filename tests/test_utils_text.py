from image_tagger.utils.text import slugify_filename


def test_slugify_filename_basic():
    assert slugify_filename("Hello World!") == "hello-world"
    assert slugify_filename("äöü 123") == "123"
    assert slugify_filename("___") == "image"


def test_slugify_filename_truncation():
    long = "verylongname" * 10
    result = slugify_filename(long, max_length=32)
    assert len(result) <= 32
