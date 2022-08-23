import time
from deep_translator import GoogleTranslator


def translate_question_words():
    texts = []
    already_processed_texts = []
    with open('research/word2vec/analogies/questions-words.txt', 'r', encoding="utf-8") as file:
        texts.extend(file.read().split("\n"))

    with open('research/word2vec/analogies/questions-words-cs.txt', 'r', encoding="utf-8") as file:
        already_processed_texts.extend(file.read().split("\n"))

    num_of_already_processed = len(already_processed_texts)

    print("TRANSLATING TEXTS...")
    translations = []
    BATCH_SIZE = 100
    batch_counter = 0
    text_batch = ""
    i = 0
    for text in texts:
        if i < num_of_already_processed:
            print("Skipping already translated.")
            pass
        else:
            print("INPUT text:")
            print(text)
            try:
                translation = GoogleTranslator(source='en', target='cs').translate((text))
                print("translation:")
                print(translation)
                translations.append(translation + "\n")
                if batch_counter == BATCH_SIZE:
                    print("Writing to file...")
                    with open('research/word2vec/analogies/questions-words-cs.txt', 'a', encoding="utf-8") as file:
                        file.writelines(translations)
                        batch_counter = 0
                        translations = []
                batch_counter = batch_counter + 1
                print(batch_counter)
            except:
                translation = "TRANSLATION ERROR"

        i = i + 1

    print("translations:")
    print(translations)


def google_translate(text_to_translate):
    translation = GoogleTranslator(source='en', target='cs').translate((text_to_translate))
    return translation


def clean_console_output_to_file():

    texts = []
    with open('research/word2vec/translations/questions-words-cs-console-copy.txt', 'r', encoding="utf-8") as file:
        texts.extend(file.read().split("\n"))
    print("text:")
    print(texts)
    texts_cleaned = []
    i = 1

    for line in texts:
        # Translation occurs in every 3rd line in current output in the format:
        """
        INPUT text:
        Athens Greece Baghdad Iraq
        translation:
        Atény Řecko Bagdád Irák
        """
        if i == 4:
            print("Adding line:")
            print(line)
            texts_cleaned.extend(line + "\n")
            i = 0
        i = i + 1

        with open('research/word2vec/translations/questions-words-cs-translated_so_far.txt', 'w+',
                  encoding="utf-8") as file:
            file.writelines(texts_cleaned)

def run_clean_console_output_to_file():
    clean_console_output_to_file()

def run_translation_of_question_words():
    translate_question_words()