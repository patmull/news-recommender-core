import time
from deep_translator import GoogleTranslator


def translate_question_words():
    texts = []

    with open('research/word2vec/analogies/questions-words.txt', 'r') as file:
        texts.extend(file.read().split("\n"))

    print("TRANSLATING TEXTS...")
    translations = []
    text_batch = ""
    for text in texts:
        print("INPUT text:")
        print(text)
        try:
            translation = GoogleTranslator(source='en', target='cs').translate((text))
        except:
            translation = "TRANSLATION ERROR"
        print("translation:")
        print(translation)
        translations.append(translation)

    print("translations:")
    print(translations)

    with open('research/word2vec/analogies/questions-words-cs.txt', 'w+') as file:
        file.writelines(translations)


def clean_console_output_to_file():

    texts = []
    with open('research/word2vec/translations/questions-words-cs-console-copy.txt', 'r', encoding="utf-8") as file:
        texts.extend(file.read().split("\n"))
    print("text:")
    print(texts)
    texts_cleaned = []
    i = 1
    for line in texts:        # Translation occurs in every 3rd line in current output in the format:
        """
        INPUT text:
        Athens Greece Baghdad Iraq
        translation:
        Atény Řecko Bagdád Irák
        """
        if i == 4:
            print("Adding line:")
            print(line)
            texts_cleaned.extend(line)
            i = 0
        i = i + 1


    print("Writing to file...")
    with open('research/word2vec/translations/questions-words-cs-translated_so_far.txt', 'w+', encoding="utf-8") as file:
        file.writelines(texts_cleaned)


clean_console_output_to_file()