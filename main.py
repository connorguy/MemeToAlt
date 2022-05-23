import logging
from urllib import request
from PIL import Image

import easyocr
from fastai.vision.all import *
import streamlit as st

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

PARAGRAPH_OCR = True


def extract_text_from_img(path_string: str):
    extracted_text = []
    # Extract text from meme
    reader = easyocr.Reader(['en'])
    result = reader.readtext(path_string, paragraph=PARAGRAPH_OCR, decoder='beamsearch', detail=0)
    extracted_text = [x.lower() for x in result]
    log.info("Extracted text:: %s" % (extracted_text))
    return extracted_text

def get_image_from_url(url: str):
    print()

if __name__ == '__main__':
    path_string = 'tmp_img'
    if os.path.exists(path_string):
        os.remove(path_string)


    img_url = st.text_input('URL of Meme?', value='https://www.bloomfieldknoble.com/wp-content/uploads/2019/03/Buzz-300x227.png')
    request.urlretrieve(img_url, 'tmp_img')
    img = Image.open('tmp_img')


    extracted_text = extract_text_from_img(path_string)

    # Clean up the output and remove potentially spurious words
    cleaned = nlp_helper.clean_text(extracted_text)
    log.info("Cleaned text:: %s" % (cleaned))
    content = ", ".join(nlp_helper.clean_text(extracted_text))

    pred, pred_idx, probs = learn_inf.predict(path_string)

    row = meme_df.loc[meme_df['index_name'] == pred]
    meme_name = row['name']

    caption = f'[Probability: {probs[pred_idx]:.02f}] ' + meme_name.values[0] + " Meme:: " + content
    log.info(caption)
    st.image(img, caption=caption, use_column_width=True)
