#!/usr/bin/env python3

import re
import sys
from unicode_codes_py3 import EMOJI_UNICODE

# See http://www.unicode.org/emoji/charts/full-emoji-list.html
# emoji_regexp = re.compile("["
#                           "\U0001F600-\U0001F64F"  # emoticons
#                           "\U0001F300-\U0001F5FF"  # symbols & pictographs
#                           "\U0001F680-\U0001F6FF"  # transport & map symbols
#                           "\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                           "\U00002702-\U000027B0"  # dingbats
#                           "\U0001F910-\U0001F93A"  # more faces
#                           "\U00002639"             # frowning face
#                           "]+")


# def remove_emoji(string):
#     return emoji_regexp.sub(r'', string)

# Will concatenate all emojis in a single string. Some side effects are that
# some emojis are composed of normal symbols, such as the # emoji (keycap_#)
# '\U00000023\U0000FE0F\U000020E3' whose first component is the HEX 0023
# corresponding to the normal # character. This causes all # to be matched.

# This is why we removed all emojis containing "keycap" in their name from
# EMOJI_UNICODE to avoid this.
emoji_pattern = '[' + ''.join(EMOJI_UNICODE.values()) + ']'
emoji_regexp = re.compile(emoji_pattern)


def remove_emoji(string):
    return emoji_regexp.sub(r'', string)


def test():
    string1 = "surprise	Why is everybody so [#TRIGGERWORD#] that I can't scooter❓❓❓ "
    string2 = "sad	I be so [#TRIGGERWORD#] when the game come on and I get ignored 😂"
    string3 = "joy	Im at the lowest point in my life.... But I can't help but be [#TRIGGERWORD#] because I still have you! 💕💍😊💑"
    string4 = "joy	I'm so [#TRIGGERWORD#] that Katie and Tom finally pulled the trigger. It was about damn time! 👫👰🏻🎊 #love #wedding #PumpRules"

    print(remove_emoji(string1))
    print(remove_emoji(string2))
    print(remove_emoji(string3))
    print(remove_emoji(string4))


if __name__ == "__main__":
    for line in sys.stdin:
        sys.stdout.write(remove_emoji(line))
