"""Script counting and the Punjabi/Urdu split, on known text."""
import io, lzma, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools_resource_count as T

ok = 0
def check(c, m):
    global ok
    assert c, m
    ok += 1

check(T.line_script("ਪੰਜਾਬੀ ਬੋਲੀ abc")[0] == "gurmukhi", "gurmukhi majority")
check(T.line_script("Србија је")[0] == "cyrillic", "cyrillic")
check(T.line_script("Srbija je")[0] == "latin", "latin")
check(T.line_script("12 34 !!")[0] is None, "no letters")
check(T.arabic_variety("ایہہ میرا گھر اے تے اوہ ساڈا دا") == "punjabi", "punjabi words")
check(T.arabic_variety("یہ میرا گھر ہے اور وہ ان کا ہے") == "urdu", "urdu words")
check(T.arabic_variety("پانی") is None, "undecided without function words")
check(T.arabic_variety("ݨ") == "punjabi", "noon with tah is Punjabi")

text = ("ਪੰਜਾਬੀ ਬੋਲੀ\n" * 3 + "ایہہ ساڈا گھر اے تے دا\n" * 2 + "یہ ہے اور کا\n" + "Srbija\n").encode()
buf = io.BytesIO(lzma.compress(text))
with lzma.open(buf) as fh:
    r = T.count_stream(fh, time.time() + 60)
check(r["complete"], "stream read to the end")
check(r["lines"] == {"gurmukhi": 3, "arabic": 3, "latin": 1}, str(r["lines"]))
check(r["arabic_lines_by_variety"] == {"punjabi": 2, "urdu": 1, "undecided": 0},
      str(r["arabic_lines_by_variety"]))
check(r["bytes_read"] == len(text), "every byte counted")
check(T.latin_variety("gde je to mesto i koje vreme") == "ekavian", "ekavian")
check(T.latin_variety("gdje je to mjesto i koje vrijeme") == "ijekavian", "ijekavian")
check(T.latin_variety("Beograd je grad") is None, "undecided")
text2 = ("gde je mesto\n" * 2 + "gdje je mjesto\n").encode()
with lzma.open(io.BytesIO(lzma.compress(text2))) as fh:
    r2 = T.count_stream(fh, time.time() + 60)
check(r2["latin_lines_by_variety"] == {"ekavian": 2, "ijekavian": 1, "undecided": 0},
      str(r2["latin_lines_by_variety"]))
c = T._Counted(io.BytesIO(b"abcdef")); c.read(4); c.read()
check(c.n == 6, "compressed bytes counted")
print(f"test_resource_count: {ok} assertions passed")
