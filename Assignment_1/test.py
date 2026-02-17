import re
import html


def pre_tokenize_clean_helper(text:str):
    # convert HTML entities into < and >
    text = html.unescape(text)
    print(text)
    # remove entire HTML tags between < >
    clean_text = re.sub(r'<[^>]+>','', text)
    # clean up white spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


print(pre_tokenize_clean_helper("&lt;A HREF=""http://www.investor.reuters.com/FullQuote.aspx?ticker=UDR.N target=/stocks/quickinfo/fullquote""&gt;UDR.N&lt;/A&gt"))