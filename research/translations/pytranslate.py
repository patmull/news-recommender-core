from translate import Translator

translator= Translator(to_lang="cs")
translation = translator.translate("Hello.")

print(translation)