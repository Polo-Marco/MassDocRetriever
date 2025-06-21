
import re

def clean_spacing(text):
    # Remove spaces before and after Chinese and ASCII punctuation
    punctuation = "，。！？：；、（）()「」『』《》【】—…·,.;:?!"
    # Remove space before punctuation
    text = re.sub(r'\s+([{}])'.format(punctuation), r'\1', text)
    # Remove space after punctuation
    text = re.sub(r'([{}])\s+'.format(punctuation), r'\1', text)
    # Remove spaces around ( and )
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'（\s+', '（', text)
    text = re.sub(r'\s+）', '）', text)
    # Optionally, collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    # Remove space after start or before end of line
    text = text.strip()
    return text
def strF2H_w_punctuation(ustring):
    # Map for full-width punctuation that isn't handled by the unicode shift
    punctuation_map = {
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '：': ':',
        '；': ';',
        '、': ',',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '「': '"',
        '」': '"',
        '『': "'",
        '』': "'",
        '《': '<',
        '》': '>',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '—': '-',
        '…': '...',
        '·': '.',
        '﹏': '_',
        '－': '-',  # full-width minus to ASCII minus
    }

    rstring = ""
    for uchar in ustring:
        # First, check for punctuation to map directly
        if uchar in punctuation_map:
            rstring += punctuation_map[uchar]
            continue
        inside_code = ord(uchar)
        if inside_code == 12288:  # full-width space
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # general full-width to half-width shift
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def strF2H(ustring):#全形轉半
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:                            # 全形空格直接轉換
            inside_code = 32
        elif 65281 <= inside_code <= 65374:   				# 全形字元（除空格）根據關係轉化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring