import logging

# from PIL import Image
import easyocr
from fastai.vision.all import *

import nlp_helper

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s:: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# Load classifier model
learn_inf = load_learner('meme-model-v1.pkl')
# load dataframe from csv, this has the references to all the trained memes.
meme_df = pd.read_csv('memes_df.csv')
meme_df['index_name'] = meme_df['name'].apply(lambda x: x.lower().replace(" ", "_"))

PARAGRAPH_OCR = False


def extract_text_from_img(path_string: str):
    extracted_text = []
    # Extract text from meme
    reader = easyocr.Reader(['en'])
    result = reader.readtext(path_string, paragraph=PARAGRAPH_OCR, decoder='wordbeamsearch', detail=0)
    extracted_text = [x for x in result]  # [x[1] for x in result] if PARAGRAPH_OCR else [x[0] for x in result]
    log.info("Extracted text:: %s" % (extracted_text))
    return extracted_text


if __name__ == '__main__':
    path_string = '/Users/connorguy/Desktop/test_memes/meme2.png'

    extracted_text = extract_text_from_img(path_string)

    # Clean up the output and remove potentially spurious words
    cleaned = nlp_helper.clean_text(extracted_text)
    log.info("Cleaned text:: %s" % (cleaned))
    content = " ".join(nlp_helper.clean_text(extracted_text))

    pred, pred_idx, probs = learn_inf.predict(path_string)
    log.info(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

    # load dataframe from csv, this has the references to all the trained memes.
    # meme_df = pd.read_csv('memes_df.csv')
    # meme_df['index_name'] = meme_df['name'].apply(lambda x: x.lower().replace(" ", "_"))

    row = meme_df.loc[meme_df['index_name'] == pred]
    meme_name = row['name']

    print(f'Probability: {probs[pred_idx]:.03f} ' + meme_name.values[0] + " meme: " + content)
