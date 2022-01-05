import re
import pandas as pd
from vncorenlp import VnCoreNLP

abbreviations = {
    'HLV': 'huấn luyện viên',
    'CĐV': 'cổ động viên',
    'THCS': 'trung học cơ sở',
    'km': ' km',
    'PGS': 'phó giáo sư',
    'GS': 'giáo sư',
    'TS': 'tiến sĩ',
    'GD-ĐT': 'giáo dục - đào tạo',
    'GD&ĐT': 'giáo dục - đào tạo',
    'GDĐT': 'giáo dục đào tạo',
    'GD': 'giáo dục',
    'ĐT': 'đội tuyển',
    'TP': 'thành phố ',
    'Tp': 'thành phố ',
    'HCM': 'hồ chí minh',
    'ĐH': 'đại học',
    'QĐND': 'quân đội nhân dân',
    'ĐHSP': 'đại học sư phạm',
    'VNDCCH': 'việt nam dân chủ cộng hòa',
    'CHXHCN': 'cộng hòa xã hội chủ nghĩa',
    'VN': 'việt nam',
    'CNGD': 'công nghệ giáo dục',
    'TSKH': 'tiến sĩ khoa học',
    'UBND': 'ủy ban nhân dân',
    'Vn': 'việt nam',
    'THPT': 'trung học phổ thông',
    'HN': 'hà nội',
    'QPAN': 'quốc phòng an ninh',
    'GDCD': 'giáo dục công dân',
    'SG': 'sài gòn',
    'FB': 'facebook',
    'CNXH': 'chủ nghĩa xã hội',
    'CNTB': 'chủ nghĩa tư bản',
    'XSTK': 'xác suất thống kê',
    'CAND': 'công an nhân dân',
    'CSGT': 'cảnh sát giao thông',
    'CSCĐ': 'cảnh sát cơ động',
    'CHLB': 'cộng hòa liên bang',
    'TBT': 'tổng bí thư',
    'TH': 'truyền hình',
    'NĐ': 'nghị định',
    'CP': 'chính phủ',
    'CC': 'chung cư',
    'CTV': 'cộng tác viên',
    'PV': 'phóng viên',
    'SN': 'sinh năm',
    'kg': ' kg',
    'TPHCM': 'thành phố hồ chí minh',
    'mm': ' mm',
    'TMCP': 'thương mại cổ phần',
    'HĐQT': 'hội đồng quản trị',
    'QLTT': 'quản lý thị trường',
    'TAND': 'tòa án nhân dân',
    'VPCC': 'văn phòng công chứng',
    'P.': 'phường ',
    'Q.': 'quận ',
    'TN-MT': 'tài nguyên - môi trường',
    'HĐXX': 'hội đồng xét xử',
    'HĐND': 'hội đồng nhân dân',
    'QL': 'quốc lộ',
    'cm': ' cm',
    'QĐ': 'quyết định',
    'TV': 'ti vi',
    'GTVT': 'giao thông vận tải',
    'CLB': 'câu lạc bộ',
    'VĐV': 'vận động viên',
    'VKSND': 'viện kiểm sát nhân dân',
    'BVĐK': 'bệnh viện đa khoa',
    'CAH': 'công an huyện',
    'HCV': 'huy chương vàng',
    'HCB': 'huy chương bạc',
    'HCĐ': 'huy chương đồng'
}


def readStopwords(fileName):
    """Hàm đọc stopword từ tập tin chứa stopword"""
    stopwords = []
    with open(fileName, encoding = 'UTF-8') as f:
        stopwords = f.read()
        stopwords = stopwords.replace(' ', '_')
    stopwords = stopwords.splitlines()
    return stopwords

stopwords = readStopwords('vietnamese_stopwords.txt')
annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')


# preprocess functions
def preprocess(text):
    text_df = pd.DataFrame([text], columns=['text'])
    text_df['text'] = text_df.apply(preprocessText, args = (abbreviations, stopwords, annotator), axis = 1)
    return text_df


def deleteLink(text):
    """Hàm xóa các đường liên kết trong văn bản"""
    deleted_links = r"(http.+[a-zA-Z0-9])|(www.+[a-zA-Z0-9])"
    text = re.sub(deleted_links, '', text)
    return text


def expandAbbreviation(text, abbreviations):
    """Hàm khai triển các từ viết tắt"""
    for key, value in abbreviations.items():
        text = text.replace(key, value)
    return text


def deleteCharacter(text):
    """Hàm loại bỏ các dấu câu, các kí tự không cần thiết"""
    deleted_characters = r",|;|!|@|#|\$|%|\^|&|\*|\(|\)|\+|\r|\n|-|`|~|\\t|<|>|\/|\?|:|'|\"|\[|\]|\{|\}|\\{1}|\||_|=|“|”|‘|’|´"
    text = re.sub(deleted_characters, ' ', text)
    dot = r"\."
    text = re.sub(dot, '', text)
    return text


def tokenizeText(text, annotator):
    """Hàm tách văn bản thành các từ"""
    word_segmented_text = annotator.tokenize(text)
    return word_segmented_text[0]


def deleteStopword(word_segmented_text, stopwords):
    """Hàm loại stopword"""
    for stopword in stopwords:
        if stopword in word_segmented_text:
            word_segmented_text = list(filter(lambda word: word != stopword, word_segmented_text))
    return word_segmented_text


def preprocessText(row, abbreviations, stopwords, annotator):
    """Hàm tiền xử lý văn bản tiếng Việt"""
    text = row['text']
    text = deleteLink(text)        
    text = expandAbbreviation(text, abbreviations)
    text = text.lower()
    text = deleteCharacter(text)
    word_segmented_text = tokenizeText(text, annotator)
    word_segmented_text = deleteStopword(word_segmented_text, stopwords)
    return word_segmented_text