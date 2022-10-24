
"""
from src.recommender_core.recommender_algorithms.content_based_algorithms.word2vec import Word2VecClass, \
    create_dictionary_from_dataframe, preprocess_idnes_corpus

word2vecClass = Word2VecClass()
# 1. Create Dictionary
create_dictionary_from_dataframe(force_update=False)
# preprocess train_corpus and save it to mongo
preprocess_idnes_corpus()



def preprocess_idnes_corpus(force_update=False):
    print("Corpus lines are above")
    cursor_any_record = mongo_collection.find_one()
    if cursor_any_record is not None and force_update is False:
        print("There are already records in MongoDB. Skipping Idnes preprocessing (1st phase)")
        pass
    else:
        path_to_pickle = 'full_models/idnes/unprocessed/idnes.pkl'
        corpus = pickle.load(open(path_to_pickle, 'rb'))
        print("Corpus length:")
        print(len(corpus))
        time.sleep(120)
        # preprocessing steps

        last_record = mongo_db.mongo_collection.find()
        print("last_record")
        print(last_record)
        print("Fetching records for DB...")
        cursor_any_record = mongo_collection.find_one()
        # Checking the cursor is empty or not
        if cursor_any_record is None:
            number_of_documents = 0
        else:

            number_of_documents = mongo_collection.estimated_document_count()
            print("Number_of_docs already in DB:")
            print(number_of_documents)

        if number_of_documents == 0:
            print("No file with preprocessed articles was found. Starting from 0.")
        else:
            print("Starting another preprocessing from document where it was halted.")
            print("Starting from doc. num: " + str(number_of_documents))

        i = 0
        num_of_preprocessed_docs = number_of_documents
        # clearing collection from all documents
        mongo_collection.delete_many({})
        for doc in generate_lines_from_mmcorpus(corpus):
            if number_of_documents > 0:
                number_of_documents -= 1
                print("Skipping doc.")
                print(doc[:10])
                continue
            print("Processing doc. num. " + str(num_of_preprocessed_docs))
            print("Before:")
            print(doc)
            doc_string = ' '.join(doc)
            doc_string_preprocessed = deaccent(preprocess(doc_string))
            # tokens = doc_string_preprocessed.split(' ')

            # removing words in greek, azbuka or arabian
            # use only one of the following lines, whichever you prefer
            tokens = [i for i in doc_string_preprocessed.split(' ') if regex.sub(r'[^\p{Latin}]', u'', i)]
            # processed_data.append(tokens)
            print("After:")
            print(tokens)
            i = i + 1
            num_of_preprocessed_docs = num_of_preprocessed_docs + 1
            # saving list to pickle evey Nth document

            print("Preprocessing Idnes.cz doc. num. " + str(num_of_preprocessed_docs))
            save_to_mongo(tokens, num_of_preprocessed_docs, mongo_collection)

        print("Preprocessing Idnes has (finally) ended. All articles were preprocessed.")

        def preprocess_question_words_file():
            # open file1 in reading mode
            file1 = open(PATH_TO_UNPROCESSED_QUESTIONS_WORDS, 'r', encoding="utf-8")

            # open file2 in writing mode
            file2 = open(PATH_TO_PREPROCESSED_QUESTIONS_WORDS, 'w', encoding="utf-8")

            # read from file1 and write to file2
            for line in file1:
                if len(line.split()) == 4 or line.startswith(":"):
                    if not line.startswith(":"):
                        file2.write(gensim.utils.deaccent(preprocess(line)) + "\n")
                    else:
                        file2.write(line)
                else:
                    continue

            # close file1 and file2
            file1.close()
            file2.close()

            # open file2 in reading mode
            file2 = open(PATH_TO_PREPROCESSED_QUESTIONS_WORDS, 'r')

            # print the file2 content
            print(file2.read())

            # close the file2
            file2.close()
            
"""