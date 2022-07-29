import json
import string
import time

import numpy as np
from gensim.utils import deaccent

from content_based_algorithms.doc2vec import Doc2VecClass
from content_based_algorithms.helper import Helper
from content_based_algorithms.lda import Lda
from content_based_algorithms.prefiller import PreFiller
from content_based_algorithms.tfidf import TfIdf
from content_based_algorithms.word2vec import Word2VecClass
from prefilling_all import run_prefilling, prepare_and_run
from preprocessing.cz_preprocessing import CzPreprocess
from data_connection import Database

def tfidf():
    tf_idf = TfIdf()
    print(tf_idf.recommend_posts_by_all_features_preprocessed(
        "chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach"))


def word2vec_method():
    word2vec = Word2VecClass()
    # random article
    database = Database()
    posts = database.get_posts_dataframe()
    random_post = posts.sample()
    random_post_slug = random_post['post_slug'].iloc[0]
    print("random_post slug:")
    print(random_post_slug)
    similar_posts = word2vec.get_similar_word2vec(random_post_slug)
    print("similar_posts")
    print(similar_posts)
    print("similar_posts type:")
    print(type(similar_posts))

    assert len(random_post.index) == 1
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    print(type(similar_posts[0]['slug']))
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0


def test_word2vec_recommendation_prefiller(database, method, full_text, reverse, random):
    prepare_and_run(database, method, full_text, reverse, random)


def try_prefillers():
    database = Database()
    method = "word2vec"
    reverse = True
    random = False
    test_word2vec_recommendation_prefiller(database=database, method=method, full_text=False, reverse=reverse, random=random)

def get_prefilled_tfidf(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_tfidf(full_text)
    return len(posts)


def get_prefilled_word2vec(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_word2vec(full_text)
    return len(posts)


def get_prefilled_doc2vec(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_doc2vec(full_text)
    return len(posts)


def get_prefilled_lda(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_lda(full_text)
    return len(posts)


def test_prefilled_recommendations():

    number_of_tfidf = get_prefilled_tfidf(full_text=False)
    print(number_of_tfidf)
    number_of_tfidf_full_text = get_prefilled_tfidf(full_text=True)
    print(number_of_tfidf_full_text)
    number_of_word2vec = get_prefilled_word2vec(full_text=False)
    print(number_of_word2vec)
    number_of_word2vec_full_text = get_prefilled_word2vec(full_text=True)
    print(number_of_word2vec_full_text)
    number_of_doc2vec = get_prefilled_doc2vec(full_text=False)
    print(number_of_doc2vec)
    number_of_doc2vec_full_text = get_prefilled_doc2vec(full_text=True)
    print(number_of_doc2vec_full_text)
    number_of_lda = get_prefilled_lda(full_text=False)
    print(number_of_lda)
    number_of_lda_full_text = get_prefilled_lda(full_text=True)
    print(number_of_lda_full_text)

    assert number_of_tfidf == 0
    assert number_of_tfidf_full_text == 0
    assert number_of_word2vec == 0
    assert number_of_word2vec_full_text == 0
    assert number_of_doc2vec == 0
    assert number_of_doc2vec_full_text == 0
    assert number_of_lda == 0
    assert number_of_lda_full_text == 0


def main():
    """
    results = []
    for i in range(0,30):
        start_time = time.time()
        tfidf()
        result = time.time() - start_time
        results.append(result)
        print("--- %s seconds ---" % result)

    print("Average time of execution:")
    print(np.average(results))
    """

    helper = Helper()
    # helper.clear_blank_lines_from_txt("datasets/idnes_preprocessed.txt")
    # doc2vec = Doc2VecClass()
    # print(doc2vec.get_similar_doc2vec("chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach", train=True, limited=False))

    # tfIdf = TfIdf()
    # print(tfidf.recommend_posts_by_all_features_preprocessed("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"))
    # word2vec = Word2VecClass()
    # word2vec.get_similar_word2vec("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")
    # tfidf()
    # word2vec = Word2VecClass()
    #word2vec.get_similar_word2vec("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")
    # word2vec.prepare_word2vec_eval()

    # word2vec = Word2VecClass()
    # word2vec.eval_wiki()

    #word2vec = Word2VecClass()
    #word2vec.save_fast_text_to_w2v()
    # word2vec.get_similar_word2vec("chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach")

    # word2vecClass = Word2VecClass()

    """
    czpreprocessing = CzPreprocess()
    list = ['kola', 'druhý', 'liga', 'zbrojovka', 'neúplný', 'tabulka', 'soutěž', 'bod', 'druhý', 'sparta', 'b', 'druhý', 'druhý', 'liga', 'sedm', 'bod', 'zbrojovka', 'uspět', 'druhý', 'liga', 'zbrojovka', 'uspět', 'v', 'třinci', 'a', 'vést', 'o', 'sedm', 'bod', 'fotbalista', 'brna', 'zvítězit', 'v', 'kola', 'druhý', 'liga', 'na', 'hřiště', 'třince', 'a', 'upevnit', 'se', 'vedení', 'v', 'neúplný', 'tabulka', 'soutěž', 'zbrojovka', 'mít', 'na', 'konto', 'bod', 'o', 'sedm', 'hodně', 'než', 'druhý', 'sparta', 'b', 'třinec', 'zůstat', 'se', 'sedm', 'bod', 'na', 'poslední', 'příčka', 'šanci', 'přiblížit', 'se', 'čelu', 'nevyužili', 'hráči', 'pražské', 'dukly,', 'kteří', 'prohráli', 'doma', 's', 'ústím', 'nad', 'labem', '0:1.', 'jsou', 'pátí', 's', '20', 'body,', 'severočeši', 'mají', 'po', 'druhé', 'výhře', 'za', 'sebou', 'o', 'čtyři', 'body', 'méně', 'a', 'jsou', 'desátí.', 'před', 'duklu', 'se', 'díky', 'lepšímu', 'skóre', 'dostala', 'na', 'čtvrté', 'místo', 'vlašim,', 'jež', 'deklasovala', 'vyškov', '5:1.', 'ještě', 'o', 'branku', 'víc', 'vstřelila', 'opava', 'a', 'po', 'triumfu', '6:1', 'nad', 'viktorií', 'žižkov', 'poskočila', 'na', 'osmou', 'příčku.', 'ztratila', 'exligová', 'příbram,', 'která', 'neudržela', 'na', 'hřišti', 'chrudimi', 'dvoubrankové', 'vedení', 'a', 'po', 'remíze', '2:2', 'je', 'třináctá.\r\nbrněnští', 'svěřenci', 'trenéra', 'richarda', 'dostálka', 'neprohráli', 'v', 'druhé', 'lize', 'již', 'devátý', 'zápas', 'po', 'sobě.', 'ten', 's', 'třincem', 'rozhodl', 'útočník', 'jakub', 'řezníček,', 'který', 'proměnil', 'pokutový', 'kop', 'a', 'potrestal', 'faul', 'brankáře', 'jiřího', 'adamušky.', 'třiatřicetiletý', 'řezníček', 'skóroval', 'už', 'poosmé', 'v', 'ročníku.', 'brno', 'vyhrálo', 'sedmé', 'z', 'posledních', 'devíti', 'utkání', 'v', 'soutěži.\r\nna', 'dukle', 'se', 'o', 'jedinou', 'branku', 'postaral', 'po', 'půl', 'hodině', 'hry', 'ústecký', 'jakub', 'emmer.', 'pražané', 'prohráli', 'čtvrtý', 'z', 'posledních', 'šesti', 'zápasů.', 'zaváhání', 'dukly', 'využila', 'vlašim,', 'která', 'se', 'po', 'výhře', '5:1', 'nad', 'vyškovem', 'vyhoupla', 'na', 'čtvrtou', 'příčku.', 'dvě', 'branky', 'zaznamenal', 'filip', 'blecha', 'a', 'vstřelil', 'v', 'posledních', 'čtyřech', 'zápasech', 'čtyři', 'góly.\r\nsouboj', 'u', 'dna', 'tabulky', 'mezi', 'chrudimí', 'a', 'příbramí', 'skončil', 'remízou', '2:2.', 'hostující', 'svěřenci', 'kouče', 'jozefa', 'valachoviče', 'vedli', 'po', 'první', 'půli', 'díky', 'brankám', 'daniela', 'procházky', 'a', 'stefana', 'vilotiče', 'o', 'dva', 'góly,', 'po', 'přestávce', 'však', 'vedení', 'ztratili.', 'mezi', '51.', 'a', '60.', 'minutou', 'se', 'o', 'vyrovnání', 'postarali', 'petr', 'rybička', 'a', 'david', 'surmaj.', 'příbram', 've', 'druhé', 'lize', 'vyhrála', 'jen', 'tři', 'z', '12', 'zápasů.\r\nopava', 'se', 'po', 'triumfu', '6:1', 'nad', 'žižkovem', 'dostala', 'do', 'středu', 'tabulky.', 'pražané', 'prohráli', 'šestý', 'z', 'posledních', 'sedmi', 'duelů', 'a', 'jsou', 's', 'osmi', 'body', 'na', 'předposlední', '15.', 'příčce.\r\ndruhá', 'fotbalová', 'liga', '-', '12.', 'kolo\r\nfk', 'fotbal', 'třinec', ':', 'fc', 'zbrojovka', 'brno', 'fc', 'zbrojovka', 'brno', '0:1', '(0:1)', 'góly:\r\ngóly:\r\n36.', 'jakub', 'řezníček', 'sestavy:\r\nadamuška', '–', 'bolf', '(90+1.', 'bedecs),', 't.', 'ba,', 'foltyn,', 'hýbl', '–', 'omasta', '(76.', 'puchel),', 'habusta', '(c),', 'šteinhübel,', 'kania', '(81.', 'javůrek)', '–', 'juřena,', 'petráň.', 'sestavy:\r\nfloder', '–', 'bariš,', 'endl,', 'štěrba,', 'jan', 'moravec', '–', 'texl,', 'm.', 'ševčík', '(90+2.', 'm.', 'sedlák)', '–', 'a.', 'fousek', '(81.', 'rogožan),', 'hladík,', 'jakub', 'řezníček', '(c)', '–', 'přichystal.', 'náhradníci:\r\nhasalík', '–', 'buneš,', 'puchel,', 'weber,', 'bedecs,', 'javůrek,', 'ntoya.', 'náhradníci:\r\nformánek', '–', 'rogožan,', 'm.', 'sedlák,', 'matula,', 'štepanovský,', 'lacík,', 'kamenský.', 'žluté', 'karty:\r\n35.', 'adamuška,', '86.', 'bolf', 'žluté', 'karty:\r\n25.', 'štěrba', 'rozhodčí:', 'ulrich', '–', 'bureš,', 'černoevič.', 'počet', 'diváků:', '293\r\nfk', 'dukla', 'praha', 'fk', 'dukla', 'praha', ':', 'fk', 'ústí', 'nad', 'labem', '0:1', '(0:1)', 'góly:\r\ngóly:\r\n31.', 'emmer', 'sestavy:\r\nf.', 'rada', '–', 'piroch,', 'j.', 'peterka,', 'kozma', '(c),', 'cienciala,', 'd.', 'souček,', 'kulhánek,', 'pázler,', 'david', 'kozel,', 'barac,', 'buchvaldek.', 'sestavy:\r\nj.', 'plachý', '–', 'hudec,', 'brak,', 'gonzalez,', 'miskovič,', 'písačka,', 'prošek,', 'ogiomade,', 'alexandr,', 'emmer,', 'čičovský.', 'náhradníci:\r\nšťovíček', '–', 'fábry,', 'ruml,', 'adediran,', 'šebrle,', 'hrubeš,', 'konan.', 'náhradníci:\r\nd.', 'němec', '–', 'mechmache,', 'm.', 'bílek,', 'cantin,', 'a.', 'černý,', 'angelozzi,', 'uličný.', 'žluté', 'karty:\r\n39.', 'buchvaldek', 'žluté', 'karty:\r\n39.', 'písačka,', '42.', 'gonzalez,', '67.', 'brak,', '77.', 'ogiomade', 'rozhodčí:', 'radina', '–', 'kotalík,', 'matoušek\r\nmfk', 'chrudim', ':', '1.fk', 'příbram', '2:2', '(0:2)', 'góly:\r\n51.', 'rybička\r\n60.', 'surmaj', 'góly:\r\n3.', 'd.', 'procházka\r\n40.', 'vilotić', 'sestavy:\r\nmikulec', '–', 'd.', 'hašek,', 'drahoš,', 'tkadlec', '(46.', 'jan', 'řezníček),', 'sokol', '–', 'v.', 'řezníček,', 'průcha,', 'd.', 'breda', '(46.', 'kesner),', 'čáp,', 'juliš', '(46.', 'surmaj)', '–', 'rybička', '(c).', 'sestavy:\r\nšiman', '–', 'sus,', 'halinský,', 's.', 'kingue,', 'vilotić', '–', 't.', 'pilík', '(c)', '(76.', 'langhamer),', 'obdržal,', 'm.', 'čermák', '(90+3.', 'voltr),', 'dedič,', 'hájek', '–', 'd.', 'procházka.', 'náhradníci:\r\nordelt', '–', 'fišl,', 'kesner,', 'jan', 'řezníček,', 'látal,', 'surmaj,', 'šplíchal.', 'náhradníci:\r\nsmrkovský', '–', 'langhamer,', 'petrák,', 'mezera,', 'e.', 'antwi,', 'voltr,', 'vávra.', 'žluté', 'karty:\r\n23.', 'd.', 'hašek,', '33.', 'drahoš,', '44.', 'tkadlec,', '49.', 'v.', 'řezníček,', '83.', 'sokol', 'žluté', 'karty:\r\n17.', 'hájek,', '63.', 'd.', 'procházka,', '70.', 'halinský', 'rozhodčí:', 'adámková', '–', 'lakomý,', 'dohnálek\r\nfc', 'sellier', '&', 'bellot', 'vlašim', ':', 'mfk', 'vyškov', '5:1', '(2:1)', 'góly:\r\n11.', 'blecha\r\n28.', 'rigo\r\n51.', 'blecha\r\n60.', 'alijagić\r\n84.', 'červenka', 'góly:\r\n43.', 'slaměna', 'sestavy:\r\nřehák', '–', 'prebsl,', 'broukal,', 'p.', 'breda,', 'v.', 'svoboda', '–', 'janda', '(66.', 'kozel),', 'křišťan,', 'višinský', '(66.', 'zinhasović),', 'blecha', '(88.', 'starý),', 'rigo', '(79.', 'červenka)', '–', 'alijagić.', 'sestavy:\r\nšustr', '–', 'fofana,', 'ilko,', 'štěpánek', '(62.', 'simr),', 'srubek', '–', 'klesa,', 'david', 'jambor,', 'slaměna', '(80.', 'tousaint),', 'němeček,', 'o.', 'vintr', '(62.', 'daouda)', '–', 'lahodný', '(62.', 'dan', 'jambor).', 'náhradníci:\r\nvágner', '–', 'zukal,', 'starý,', 'červenka,', 'kozel,', 'zinhasović,', 'hošek.', 'náhradníci:\r\nspurný', '–', 'hanuš,', 'simr,', 'cabadaj', '(c),', 'dan', 'jambor,', 'tousaint,', 'daouda.', 'žluté', 'karty:\r\nžluté', 'karty:\r\n19.', 'němeček,', '57.', 'srubek', 'rozhodčí:', 'proske', '–', 'dobrovolný,', 'pochylý', 'počet', 'diváků:', '350\r\nsfc', 'opava', ':', 'fk', 'viktoria', 'žižkov', '6:1', '(2:1)', 'góly:\r\n12.', 'darmovzal\r\n32.', 'lukáš', 'holík\r\n59.', 'kadlec\r\n68.', 'pikul\r\n85.', 'kramář\r\n86.', 'kramář', 'góly:\r\n25.', 'batioja', 'sestavy:\r\ndigaňa', '–', 'celba,', 'hnaníček', '(c),', 'janoščín', '(75.', 'gorčica),', 'kadlec', '–', 'darmovzal,', 'rychlý', '–', 'pikul', '(70.', 'kramář),', 'didiba,', 'helešic', '(81.', 'šcudla)', '–', 'lukáš', 'holík', '(70.', 'rataj).', 'sestavy:\r\np.', 'soukup', '–', 'koželuh,', 'š.', 'gabriel,', 'súkenník', '(c),', 'řezáč', '–', 'žežulka', '(46.', 'sakala)', '–', 'diamé', '(75.', 't.', 'zeman),', 'sixta', '(75.', 'kytka),', 'd.', 'richter', '(46.', 'nabijev),', 'muleme', '–', 'batioja', '(83.', 'petrlák).', 'náhradníci:\r\nlasák', '–', 'pisačič,', 'kramář,', 'rataj,', 'gorčica,', 'helebrand,', 'šcudla.', 'náhradníci:\r\nšvenger', '–', 'nabijev,', 'kop,', 'petrlák,', 't.', 'zeman,', 'kytka,', 'sakala.', 'žluté', 'karty:\r\n6.', 'hnaníček,', '55.', 'lukáš', 'holík,', '74.', 'rataj', 'žluté', 'karty:\r\n37.', 'žežulka,', '49.', 'muleme,', '90+1.', 'sakala', 'rozhodčí:', 'klíma', '–', 'šimáček,', 'slavíček\r\nsk', 'líšeň', '2019', ':', 'fk', 'varnsdorf', '1:0', '(1:0)', 'góly:\r\n33.', 'silný', 'góly:\r\nsestavy:\r\nveselý', '–', 'pašek,', 'jeřábek,', 'o.', 'ševčík', '(c),', 'lutonský', '–', 'otrísal', '–', 'bednář', '(77.', 'stáňa),', 'málek,', 'matocha,', 'rolinek', '(86.', 'burda)', '–', 'silný', '(86.', 'ulbrich).', 'sestavy:\r\na.', 'richter', '–', 'heppner,', 'hušek', '(59.', 'šimon),', 'kouřil,', 'žák', '–', 'bláha,', 'm.', 'richter', '(85.', 'kocourek),', 'rudnytskyy', '(c),', 'ondráček', '(73.', 'lauko),', 'zbrožek', '–', 'dordić.', 'náhradníci:\r\nvítek,', 'marek', '–', 'stáňa,', 'chwaszcz,', 'ulbrich,', 'burda,', 'zúbek.', 'náhradníci:\r\nvaňák', '–', 'pajkrt,', 'lauko,', 'šimon,', 'velich,', 'kocourek,', 'm.', 'kubista.', 'žluté', 'karty:\r\n20.', 'otrísal,', '39.', 'jeřábek,', '58.', 'silný,', '64.', 'rolinek,', '71.', 'valachovič,', '90.', 'veselý', 'žluté', 'karty:\r\n18.', 'rudnytskyy,', '84.', 'šimon', 'červené', 'karty:\r\n69.', 'otrísal', 'červené', 'karty:\r\n31.', 'rudnytskyy', 'rozhodčí:', 'krejsa', '–', 'pfeifer,', 'poživil', 'počet', 'diváků:', '580\r\n2.', 'liga', 'klub', 'z', 'v', 'r', 'p', 's', 'b', '1.', 'brno', '12', '9', '2', '1', '25:11', '29', '2.', 'sparta', 'b', '12', '7', '1', '4', '23:14', '22', '3.', 'líšeň', '11', '6', '3', '2', '18:11', '21', '4.', 'vlašim', '12', '6', '2', '4', '27:17', '20', '5.', 'dukla', 'praha', '12', '6', '2', '4', '19:14', '20', '6.', 'prostějov', '11', '6', '1', '4', '13:15', '19', '7.', 'varnsdorf', '11', '5', '3', '3', '21:16', '18', '8.', 'opava', '12', '4', '4', '4', '17:15', '16', '9.', 'táborsko', '11', '5', '1', '5', '11:12', '16', '10.', 'ústí', 'nad', 'labem', '12', '4', '4', '4', '13:16', '16', '11.', 'jihlava', '12', '4', '3', '5', '11:15', '15', '12.', 'vyškov', '12', '4', '2', '6', '21:19', '14', '13.', 'příbram', '12', '3', '4', '5', '16:21', '13', '14.', 'chrudim', '12', '2', '3', '7', '10:19', '9', '15.', 'žižkov', '12', '2', '2', '8', '13:24', '8', '16.', 'třinec', '12', '2', '1', '9', '9:28', '7']
    doc_string = ' '.join(list)
    print(deaccent(czpreprocessing.preprocess(doc_string)))
    """
    # word2vecClass = Word2VecClass()
    word2vec_method()
    try_prefillers()
    # 1. Create Dictionary
    # word2vecClass.create_dictionary_from_dataframe(force_update=False, filter_extremes=False)
    # preprocess train_corpus and save it to mongo

    # word2vecClass.preprocess_idnes_corpus()

    # word2vecClass.preprocess_idnes_corpus()
    # word2vecClass.eval_idnes_basic()
    # word2vecClass.remove_stopwords_mongodb()
    # run_prefilling()
    # lda = Lda()
    # print(lda.get_similar_lda('chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach'))


if __name__ == '__main__':
    main()
