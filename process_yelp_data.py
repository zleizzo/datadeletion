import json
import time
import random
import string
import numpy as np
import pickle


def process_yelp_data(path='data/review.json', d=1500, verbose=True):
    t1 = time.time()

    random.seed(0)

    # These users were selected because:
    # 1. They each have between 1 & 20 reviews in the dataset; in particular, there are 10 users with 1 review, 10 users with 2 reviews, etc.
    # 2. In total, they provide a roughly balanced mix of positive & negative reviews (approximately 55% positive & 45% negative).

    users = set(['glgY1PfrSbYcDdAwVPP1Ag','e16L4L_XZhWd-dtamyIcgQ','xkw9mDmLQ0qUeFvOKy8GsA','wV-zAB1wYQU-x-akrPwgyA',
                 'Z242MNoiMvS3J2XJTaFTag','yOj-EsTaL0gAPsBPZAsBiw','P_j0to2l-pQHqGsz5rP0-g','Qa1m8sPki2DRWVW7L12q6g',
                 'On40-KK9SSpWXUa-7cxqDQ','kFuy5qf9Wh0VHeagCfIpYQ','aPhwq_YYD1uCO3Q1mSlzcQ','W-rX3Mg9QCXlp2I5k8uZZw',
                 '9qeb4Z0BfIJ9Uz62TR2foQ','PX73oUx1x2eMcrphjoBz_Q','0fyh5fWen3sUrpaq07j2QQ','Gs9MUwjkKGGJeOv17dfEnA',
                 'nTYDaq7Irzz8dxPFyaMJwA','uc5gDU1IBaYqRkbiSRLqVQ','O_HuTyME8qXCcScaVptHMQ','VkCFXbUrkggd6UMkPGaoTA',
                 'hDMMb0wj5ZsUlJhLDX8DMQ','Aiy2pyA2d_tx3VhF3UDXeA','8u40NmUttgpEpAJr249Fcg','PkeTjQ_wrhZ-qtLjL_dUmg',
                 'Qss-8g6xgZZmiXXxlfbLcQ','eB8yycBC6IIES0IslsX71g','ltkgihu7rRNwyLGmvzQarQ','ohkA3vhJ3X2XSVY8dJ3qUw',
                 'u4ZuffEY2NTWhRzpeFN7OQ','qzpHul1cL7nf8f0OYXwz3Q','z7Fd9GM8_fESYUSFSKCvGQ','Z1yy2pa7G3JB7JWJfmeg2A',
                 'UguHPPJrJFWxppRekSP4gg','Sx-gnCv1NaYyKoRGNrGYFg','aRL6Xt7b6aKtJGKErvJBRw','1_VdegJ-xiltPO1xn3-H7w',
                 'S67h3-E9SYvXcp_6fqlBKg','7YxtCHYSvaE9UzauY0q1uw','E_z8k5o93yGLdHnKhbz9yw','HJ9SFDXCns0Z_La0hxLi4g',
                 'Lp48M5Zsc7USVWm9nSefMg','5m7V5T9azl-W6tGX8UKtlw','Lsbcp7XZnL3Wz418Z4xsIw','aA4eTEx2XlLgZQe72_EF1w',
                 'PkGyUoDiDAyP901CecQT_A','QRfBGDwTsu8Ci_70j_0A0w','iDfQikdiXnhu03VBFM4rqg','4d648Hu4zu1gUqR34PjAqw',
                 'zKFRpOiGqNO9EZX6vO32qw','LEjU3bnSS0_qpdMy4a69oQ','0IgI0NBdRxtQOrYFmrAQlA','he6N1p114f1U6LnL4D5UZw',
                 'ZfaJZpjn5JNiLUZputC5lQ','NrnaYqXZsWzO1maOiwurhQ','LPMBPzPJw7JZM-uHrNA8yA','9SzSlEGovHa7py8fkOvrWQ',
                 'hVPbm9wSo7Rp66ThDCXkWA','yEJqcuknyIYxJmwC0qPpeA','_vCfMFECvN46kEqm5cuNNg','qhYMp7KJGiwJ_r2wl1VdBg',
                 '8RtTguVRzQFO0U0vyo7OKw','5nzvGSXHzraVAS5qIgdTIw','KWW8hgUG_XZNK6jlkgB-sA','cmRGAidUbW9aHs3Jm7dE5Q',
                 'NFwDU_yqec7_8_Qg_6Lmpw','win1W7bsofdOSNRxgm85fQ','c8FnrtMcp4iee6rSpWLBOg','tfzOPdndTplj7FyTkgLR4A',
                 'dtAUgabgols-teWseuT7cA','3rjbOR9Q2vWlxXiPyU61eg','pnXirbkfP8dAQMJt6WA3Wg','4RMYjhLUqMzwqjWjcrpiew',
                 'dp5uv7y7KCzsIFXnWsDrwA','Hw9OyG9cx-XlAeqhzL7Jlw','qNWR05ELdRP1PWdD_rHB2g','D4R85SMViv3ecq27qQu_Mg',
                 'gL1ASefklu30ENtldQg2OA','4FVFfff0dS2F6lZAVw-n-g','IRPe0q29IgkMxuFlC_AuFQ','yAlzGSBymSE2517AWDTT8A',
                 'dUVCTAynXUYc0pmnmCXF4w','ZH6QGYWHMsZNtdztX0tbTQ','2N4VBte7Pn9joYiGSj21QQ','-n7PJW8rEciRWyvStnG4hg',
                 'LrqHY2HzP9yAFrbR__vi4w','8euDKmVMRQfLRVh9f5HSzw','B19GhPyFxVT87-33_Z5gLQ','f4Ly2POl_bZiXp04xyX5TA',
                 'XqRGBt7_qNVPA-sQMvpneg','fwZZucynx0ugwJQXCmiKLQ','wenFKNoaIxopNCYfmWODzQ','_fzLZc1mycYPKoGn8nNqvw',
                 'bHA6QAU80u_PFU6NJ-WZxw','539TneaQp-AoSTblKwVLZg','Trhje4mgDLEuFtX6eDuQeA','4ZvmICRMtp8FcYdt94vG7w',
                 'Wgn8KqNkpLNKBbtW9ksQNg','CAKxM2KedvSOzk3huL2M8w','WpP8mkVAAi2g2UKUSSgR-Q','XL65mTw2s3-FpUWvCf2cwg',
                 'YX14lFP2JgSytsZSxYkFQw','i55P5vrrGhjIHe1ZA_C-Lw','Gt1HHcQQ6sNjadqF2yvDwA','f1H3PORZyXp_VZTHzL49Pw',
                 'IlxExAP_rJci4wZ-8lqWSA','XFaFWxjCr1_eyUyctA8U0g','gxryro-W_O6Df1N_TMS3Bw','fixnC0Gnxi8PfBwRsQiP5A',
                 'WOXtbPwnwrFIt8f8E-UGNA','KQbr-p40xTdlwpkvgL-Kdg','_DNfpRB0faPfJ9RCLuVXIQ','6rG3-v-x45q1CPZ2zn2LkA',
                 'lNsakXXHEGFtpxbELs7crg','egdHgW87SbC7BmJsm-qZEw','yJFET-9-u3DTu47mTpC45Q','Aj_ObA3dPjqdsYPy-Ysc6A',
                 'T20AK5rz8XvvekQwXcFqUA','abD8lKTNrb6Y1g3RhZbwEw','SQOV0J4ZnyGMWbHplgzoWw','eXj__DVBun03Lg9CcQZpBA',
                 'e9cFYU_MiQ_FEP0gCizoHw','r0LoyNhlG3Vowh9_vVtB3Q','vPPpx4xwzxlzn9pCU6j2rw','dH932JzCjhu0OUH5SVaoBQ',
                 'wwiPgEfZmY8E6fgJZ4u4-g','SyBQop3HHn_SWFKcgO3cFg','sMVwCjsp1qgBhC_5gSlKWA','dmBJyk_AS6YRjB4zKVXd4w',
                 '8ZZdPK41e-ImxfQEpfIEeA','qwQYdnLol8YYEPkM4H4v6Q','Z4VF-tv5ibkhv_kzVny0NA','w52Ge6xsH7_2DGJqhLJb6A',
                 '1AJD4IldxrJyjjRJcZneUg','A05EsmerAhfpwNiYNJ7PfA','jomwn5RhvThGlzsQFXffzw','kHN2Lv1dfGSTbzGiygiEew',
                 'z2BmANbsGM3Fr0rnlDRnjA','1CtBdnoJSrlp3YhTf9sC8g','LPMbxXqyQblDiNRoxRCRbA','_L4S9fGvvO2801KBCYHVYA',
                 'rWTuadggnkxGyeNqIvJ2Mg','6h8LPec7jWlwiPMuQ33RXg','iW8cBWtZWaLSe3Q-vPxGTQ','WmKkDtbINzCLeS1DUE40MA',
                 'O7QToGjR1XokgiEWC64ivQ','-8eWskzEFJf9fN21Eo77vw','7J8UAcGqJ3R6A-fMKNocXw','L0lzh2yW4348mjteq45zlw',
                 '3PaNIWP2uFZBcayFa1yUkQ','mpxxmNU3coEWcpU8cpDeMg','BxjajkJiWvyOAdLxXSjGOA','-Fz5csjvIubCn1LXDW6WcQ',
                 'eb148dnSmV0kXEnIi6uByA','-f0hp6yAYrcqxTNnPXzmyA','5WuO2GStwA2IdBnOnGeLbA','WSjmD3hVdnJ3sKEzrOWrZQ',
                 'ENkFp0Zv7EPt5P5UGbJYpA','rhuF_oUmqKYB0j02gJAEKg','GNkV5m-X1ZH72oaYp4Uf3g','jXBYwJM0w-RbWrmuwz7MzA',
                 'CpFXpttsvS1r8gnZkPlJHw','nYJNx2KHs9m9AnsipEXsPA','US69KHwBwNoNerqhS_Iukg','gskndoX2U6VPDNsvoFFITg',
                 'ZxcohiEeUvy2QzkAIKJwuQ','ta3y9x2nmMJOXfB6SWT_0A','6BhENyBfpdXmX8QDICM92w','AJ9S6hw8hc6kEFL47DTLaQ',
                 'Sha9wED_o6ZEeYQFSFLfxg','6Mm4n2a3OHMOFgWXLbOiPQ','b7U-ZMom4C0F3r59w0Ny5g','ooR9xYrqKjg7jBensoVOUw',
                 'j-H4_6hRIS5evd3fbtw1Bw','e-LlqWVL9eUkTm9H9riYHQ','OWk9jdNK2Qy9Ziu9VTT69A','AJQYfzSO7EfHq1czFFcGKQ',
                 'krRX49W5e9HeCEZhEvRnfA','ge_yqJd241NMPuyOS_J_LA','ixwQ0iEyEmMPyI5hTUuKfw','ZWNcfO-JX93PsDa1cn3OKw',
                 'ZTkuKoOt2vp584A-8hDvBA','7cR5xBHckQXkxQgG50d2Wg','X02y9uYh4l5Ow5YGXP3Nfw','vZXeLngW20Km0NmiWwFBDg',
                 'M4a0gqJ4mhM6bUV_y-Gkcg','ibRTPCNETk1DIUku4Ccc8g','hwPtLFdWie49m2cZzdD0Xw','rx78xr5_zPxFX6QdLFJv7g',
                 'tIzUt63MYMKza-hP680RxQ','55GPi56Lrr0a3_eomcKK6g','O_XLrORc2y54s7EfgAWlQQ','DQxU4yyFyMg5Yctlsh3kLA',
                 'SyzePMH0zoO0UDPRLVFk2A','j2Opul0jCCBmUZVH3oHQog','VU-0QucMwlXiKUXJpNvBgw','qUCpe68BpojNmzDopIgPoA',
                 'eqUAMhR5YxpD_J-6AuPolw','qhTa_gzOMAlVuIr1kvQNKg','MWoFseXWgqG4qu-B3Y8qlg','x9yIvzu0m81Eq-6cHCanQg'])

    # Now load the data from the downloaded files.
    # Separate the reviews that corresponds to the selected users from the rest.

    with open(path, 'r') as review_file:
        other_reviews = []
        dataset = []
        for line in review_file:
            review = json.loads(line.rstrip())
            if review['user_id'] in users:
                dataset.append({
                    'user_id': review['user_id'],
                    'stars': review['stars'],
                    'text': review['text'],
                })
            else:
                other_reviews.append(review['text'])

    # Sample 1000 reviews from the set of reviews not corresponding to the selected users.
    # Use these reviews to create a vocabulary set of 500 words.

    def extract_words(text):
        text = text.lower()
        exclude = set(string.punctuation + string.digits)
        text = ''.join(ch for ch in text if ch not in exclude)
        words = text.split()
        words = filter(lambda w: len(w) > 1, words)
        return list(words)

    review_sample = random.sample(other_reviews, 5000)
    review_sample = [extract_words(r) for r in review_sample]

    vocabulary = set([])
    for review in review_sample:
        for word in review:
            vocabulary.add(word)

    counts = {}
    for word in vocabulary:
        counts[word] = 0
    for review in review_sample:
        for word in vocabulary:
            if word in review:
                counts[word] += 1 # count number of reviews in which word appears

    vocabulary = sorted(vocabulary, key=lambda w: counts[w], reverse=True)[:d]

    # Now build the final dataset, representing each review with a bag of words representation

    def txt_to_vec(text, vocabulary):
        words = extract_words(text)
        d = len(vocabulary)
        v = np.zeros(d)
        for i in range(d):
            for word in words:
                if word == vocabulary[i]:
                    v[i] += 1
        return v

    n = len(dataset)
    d = len(vocabulary)
    X = np.zeros((n,d))
    y = np.zeros(n) - 1 # for {-1, 1} labels
    user_id = []
    for idx in range(n):
        review = dataset[idx]
        user_id.append(review['user_id'])
        X[idx] = txt_to_vec(review['text'], vocabulary)
        if review['stars'] >= 4:
            y[idx] = 1

    t2 = time.time()
    if verbose:
        print('Total time to load yelp dataset: {} minutes'.format((t2 - t1)/60))

    return {
        'X': X,
        'y': y,
        'user_ids': user_id,
    }

def save_yelp_data(filename='data/yelp.pkl'):
    data = process_yelp_data()
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print('Data saved to {}.'.format(filename))

def load_yelp_data(filename='data/yelp.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
