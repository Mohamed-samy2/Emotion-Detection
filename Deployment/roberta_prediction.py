import spacy
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline

roberta_tokenizer = RobertaTokenizer.from_pretrained(r"D:\Cognitive project\Roberta Transformer\Tokenizer")

roberta_model = RobertaForSequenceClassification.from_pretrained(r"D:\Cognitive project\Roberta Transformer\Model")

class roberta:
    def __init__(self, text):
        self.text = text

    def predict(self):
        nlp = spacy.load("en_core_web_sm")
        negative_words = ["not", "n't", "nt"]
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        doc = nlp(self.text)

        lemmatized_tokens = []

        for token in doc:
            # Check if the token is a stop word or a negative word
            if token.text.lower() in stop_words or token.text.lower() in negative_words:
                # If it is a stop word or negative word, append the token as it is to the list
                lemmatized_tokens.append(token.text)
            else:
                # If it is not a stop word or negative word, append the lemma of the token to the list
                lemmatized_tokens.append(token.lemma_)

        text = " ".join(lemmatized_tokens)
        bert_nlp = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)
        output = bert_nlp(text)

        return output[0]['label']
