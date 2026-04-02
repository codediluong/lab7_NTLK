# lab7_nltk.py

import random
import re
import string
import nltk
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')

# =========================
# DOWNLOAD TÀI NGUYÊN CẦN THIẾT
# ========================= 
packages = [
    "gutenberg",
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet",
    "omw-1.4",
    "names",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]

for pkg in packages:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

from nltk.corpus import stopwords, wordnet, names
from nltk.book import *
from nltk.metrics import edit_distance


# =========================
# 1. Liệt kê tên các corpus
# =========================
def exercise_1():
    print("\n===== 1. TEN CAC CORPUS =====")
    try:
        print(nltk.corpus.__dir__()[:50])  # in thử 50 corpus đầu
    except Exception as e:
        print("Loi:", e)


# ==========================================
# 2. Liệt kê danh sách stopword nhiều ngôn ngữ
# ==========================================
def exercise_2():
    print("\n===== 2. STOPWORDS THEO NGON NGU =====")
    langs = stopwords.fileids()
    print("Cac ngon ngu ho tro:", langs)
    for lang in langs[:10]:
        print(f"\nNgon ngu: {lang}")
        print(stopwords.words(lang)[:20])


# ====================================================
# 3. Kiểm tra danh sách stopword bằng các ngôn ngữ khác nhau
# ====================================================
def exercise_3():
    print("\n===== 3. KIEM TRA STOPWORD =====")
    word_to_check = "the"
    langs = stopwords.fileids()

    for lang in langs[:10]:
        is_stopword = word_to_check in stopwords.words(lang)
        print(f"'{word_to_check}' co phai stopword trong '{lang}'? -> {is_stopword}")


# ==========================================
# 4. Loại bỏ stopword khỏi một văn bản cho trước
# ==========================================
def remove_stopwords_from_text(text, language="english"):
    tokens = nltk.word_tokenize(text)
    sw = set(stopwords.words(language))
    filtered = [w for w in tokens if w.lower() not in sw]
    return filtered

def exercise_4():
    print("\n===== 4. LOAI BO STOPWORD KHOI VAN BAN =====")
    text = "This is a simple example to show how to remove stopwords from a sentence."
    filtered = remove_stopwords_from_text(text, "english")
    print("Van ban goc:", text)
    print("Sau khi loai stopword:", filtered)


# ==========================================
# 5. Bỏ qua stopword từ danh sách stopword
# ==========================================
def exercise_5():
    print("\n===== 5. BO QUA STOPWORD TU DANH SACH =====")
    words_list = ["this", "is", "an", "apple", "and", "banana", "the", "fruit"]
    sw = set(stopwords.words("english"))
    result = [w for w in words_list if w.lower() not in sw]
    print("Danh sach goc:", words_list)
    print("Sau khi bo qua stopword:", result)


# ===================================================
# 6. Tìm định nghĩa và ví dụ của một từ bằng WordNet
# ===================================================
def exercise_6():
    print("\n===== 6. DINH NGHIA VA VI DU BANG WORDNET =====")
    word = "computer"
    synsets = wordnet.synsets(word)

    if not synsets:
        print("Khong tim thay tu trong WordNet.")
        return

    for i, syn in enumerate(synsets[:5], 1):
        print(f"\nNghia {i}: {syn.name()}")
        print("Dinh nghia:", syn.definition())
        print("Vi du:", syn.examples())


# ==========================================
# 7. Tìm từ đồng nghĩa và trái nghĩa
# ==========================================
def exercise_7():
    print("\n===== 7. TU DONG NGHIA VA TRAI NGHIA =====")
    word = "good"
    synonyms = set()
    antonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())

    print("Tu dong nghia:", sorted(synonyms)[:30])
    print("Tu trai nghia:", sorted(antonyms))


# ===========================================================
# 8. Tổng quan về tag, chi tiết 1 tag cụ thể, dùng regex
# ===========================================================
def exercise_8():
    print("\n===== 8. TAG VA REGEX =====")
    sentence = "The quick brown fox jumps over the lazy dog."
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    print("Gan nhan tu loai:")
    print(tagged)

    # Thống kê tag
    tags = [tag for word, tag in tagged]
    print("Cac tag xuat hien:", set(tags))

    # Lọc tag bắt đầu bằng NN (danh từ) bằng regex
    noun_tags = [pair for pair in tagged if re.match(r"NN.*", pair[1])]
    print("Cac tu co tag dang NN.* :", noun_tags)


# ==========================================
# 9. So sánh độ giống nhau của 2 danh từ
# ==========================================
def exercise_9():
    print("\n===== 9. SO SANH HAI DANH TU =====")
    word1 = "car"
    word2 = "automobile"

    syn1 = wordnet.synsets(word1, pos=wordnet.NOUN)
    syn2 = wordnet.synsets(word2, pos=wordnet.NOUN)

    if syn1 and syn2:
        similarity = syn1[0].wup_similarity(syn2[0])
        print(f"Do giong nhau giua '{word1}' va '{word2}':", similarity)
    else:
        print("Khong tim thay synset phu hop.")


# ==========================================
# 10. So sánh độ giống nhau của 2 động từ
# ==========================================
def exercise_10():
    print("\n===== 10. SO SANH HAI DONG TU =====")
    word1 = "run"
    word2 = "walk"

    syn1 = wordnet.synsets(word1, pos=wordnet.VERB)
    syn2 = wordnet.synsets(word2, pos=wordnet.VERB)

    if syn1 and syn2:
        similarity = syn1[0].wup_similarity(syn2[0])
        print(f"Do giong nhau giua dong tu '{word1}' va '{word2}':", similarity)
    else:
        print("Khong tim thay synset phu hop.")


# ==========================================================
# 11. Đếm số lượng tên nam/nữ và in 10 tên đầu tiên
# ==========================================================
def exercise_11():
    print("\n===== 11. TEN NAM VA NU =====")
    male_names = names.words("male.txt")
    female_names = names.words("female.txt")

    print("So luong ten nam:", len(male_names))
    print("So luong ten nu:", len(female_names))

    print("10 ten nam dau tien:", male_names[:10])
    print("10 ten nu dau tien:", female_names[:10])


# ==========================================================
# 12. In 15 kết hợp ngẫu nhiên đầu tiên gắn nhãn nam/nữ
# ==========================================================
def exercise_12():
    print("\n===== 12. 15 TEN NGAU NHIEN GAN NHAN =====")
    labeled_names = ([(name, "male") for name in names.words("male.txt")] +
                     [(name, "female") for name in names.words("female.txt")])

    random.shuffle(labeled_names)
    print(labeled_names[:15])


# ==========================================================
# 13. Trích ký tự cuối cùng và tạo mảng mới (chữ cuối, nhãn)
# ==========================================================
def exercise_13():
    print("\n===== 13. KY TU CUOI CUNG VA NHAN =====")
    labeled_names = ([(name, "male") for name in names.words("male.txt")] +
                     [(name, "female") for name in names.words("female.txt")])

    features = [(name[-1].lower(), gender) for name, gender in labeled_names]
    print("15 phan tu dau tien:")
    print(features[:15])


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()
    exercise_7()
    exercise_8()
    exercise_9()
    exercise_10()
    exercise_11()
    exercise_12()
    exercise_13()