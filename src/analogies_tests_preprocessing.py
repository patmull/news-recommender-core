import collections


def double_words(sentence):
    words = sentence.split()
    word_counts = collections.Counter(words)
    list_of_duplicate_counts = []
    for word, count in sorted(word_counts.items()):
        print('"%s" is repeated %d time%s.' % (word, count, "s" if count > 1 else ""))
        list_of_duplicate_counts.append(count)
    return list_of_duplicate_counts


with open('../research/word2vec/analogies/questions-words-cs-preprocessed.txt', mode="r") as f:
    with open('../research/word2vec/analogies/questions-words-cs.txt', mode="w") as f2:
        for line in f:
            print(line)
            duplicated_counts = double_words(line)
            print(duplicated_counts)
            if all(i == 1 for i in duplicated_counts):
                f2.write(line)


