# import string
# from alibi.utils.lang_model import *
#
#
# def functionalities(lm: LanguageModel, text):
#     stopwords = ['and', 'the', 'but', 'a', 'this']
#
#     tokens = lm.tokenizer.tokenize(text)
#     string_tokens = lm.tokenizer.convert_tokens_to_string(tokens)
#
#     print("Tokens:", tokens)
#     print("String:", string_tokens)
#     print("Ids:", lm.tokenizer.convert_tokens_to_ids(tokens))
#
#     stopwords = [token for i, token in enumerate(tokens) if lm.is_stop_word(
#                                                             text=tokens,
#                                                             start_idx=i,
#                                                             punctuation=string.punctuation,
#                                                             stopwords=stopwords)]
#     print("Stopwords:", stopwords)
#
#     punctuation = [token for token in tokens if lm.is_punctuation(token, string.punctuation)]
#     print("Punctuation:", punctuation)
#
#
# if __name__ == "__main__":
#     text = "this and this this is a first sentence !?! but nothing afterwards don't matter..."
#
#     print("\n\n RobertaBase \n =============== \n")
#     lm = RobertaBase()
#     functionalities(lm, text)
#     del lm
#
#     print("\n\n BertBaseUncased \n =============== \n")
#     lm = BertBaseUncased()
#     functionalities(lm, text)
#     del lm
#
#     print("\n\n DistilbertBaseUncased \n =============== \n")
#     lm = DistilbertBaseUncased()
#     functionalities(lm, text)
#     del lm
