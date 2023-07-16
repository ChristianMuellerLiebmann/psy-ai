import spacy

def load_spacy_model():
    return spacy.load("de_core_news_sm")

def censor_personal_data(text, nlp):
    doc = nlp(text)

    censored_chars = []
    for ent in doc.ents:
        if ent.label_ in ["PER", "LOC", "GPE", "FAC", "ORG"]:
            start = ent.start_char + 1
            end = ent.end_char
            censored_chars.extend(range(start, end))

    censored_text = "".join([char if i not in censored_chars else "." for i, char in enumerate(text)])

    return censored_text

nlp = load_spacy_model()
censored_text = censor_personal_data(text, nlp)
