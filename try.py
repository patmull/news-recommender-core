
from content_based_algorithms.doc2vec import Doc2VecClass

searched_slug = 'k-pocte-zbran-hradni-straz-vyzaduje-preciznost-odolnost-i-bojovy-um'
doc2vec = Doc2VecClass()
print(doc2vec.get_similar_doc2vec(searched_slug))
