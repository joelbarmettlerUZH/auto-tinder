import urllib.request
import os

url = "https://images-ssl.gotinder.com/u/p8fK1grrPhxnWsvxFezVnL/3ByvN6Niw8zmyU4XMnxbaX.jpeg?Policy=eyJTdGF0ZW1lbnQiOiBbeyJSZXNvdXJjZSI6IiovdS9wOGZLMWdyclBoeG5Xc3Z4RmV6Vm5MLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2MjMyNTEwMTR9fX1dfQ__&Signature=LxQKkIcWv2FxVJBWPfw83RaP24E79sjWHawnhn1epdRVc02gE-uv1wX41wxM~m~VSiC4Le5nvGJpAh~AHY3qj1mUQkSr-51AXSgaqC~U6SpcY1XRm~nI0bwaPDWfiD6CKAAyv4mRFthnWD7lHPb76k9eIGVlV~vKo6yyUOztD8nzr~~DVP1Sd6C78lIsws1EqLTjZCweqou2P4v8CGLJGIt1zra5bqrs35prwv36-KSeq~WK1~cQDsKqNF2URiamB6-QyjThY4~SyArgnSgv8OwQeJelUcxYawRKSEhIypdCMFdcuvgOI-KSUd-N5qJbY2P6Hq~yhZ~ld~wB32NVLA__&Key-Pair-Id=K368TLDEUPA6OI"
localpath = f"/home/michael/Developement/dating-robot/auto-tinder-images/{1}.jpg"
is_existed =    os.path.isfile(localpath)
urllib.request.urlretrieve(url,localpath)




