import math
from collections import Counter
import pandas as pd
from flask import Flask, request, jsonify
import nltk as nltk
import numpy as np
from pathlib import Path
import pickle
from google.cloud import storage
import inverted_index_gcp
from nltk.corpus import stopwords
import re
import BM_25_from_index as BM25
from collections import defaultdict


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

############################# Helper Methods #############################

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def get_content_from_storage(bucket, file_name):
    """
    Args:
        bucket: string: name of bucket in GCP.
        file_name: string: path of file in bucket in GCP to get.
    Returns:
        the file loaded from storage.
    """
    blob = storage.Blob(f'{file_name}', bucket)
    with open(f'./{file_name}', 'wb') as file_obj:
        blob.download_to_file(file_obj)
    with open(Path("./") / f'{file_name}', 'rb') as f:
        return pickle.load(f)


def tokenize(text):
    """
    Tokenize text by RE and stopwords.
    Args:
        text: string: text to tokenize.
    Returns:
        string: text after tokenization.
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens_without_stop_words = [tok for tok in tokens if tok not in english_stopwords]
    return tokens_without_stop_words


def get_index_from_bucket(bucket, index_name):
    """
    Args:
        bucket: string: name of bucket in GCP.
        index_name: string: path of index pkl file in bucket in GCP to get.
    Returns:
        InvertedIndex: an inverted index class instance read from the pickle file loaded.
    """
    blob = storage.Blob(f"postings_gcp/{index_name}.pkl", bucket)
    with open(f'./{index_name}.pkl', 'wb') as f:
        blob.download_to_file(f)
    index = inverted_index_gcp.InvertedIndex.read_index('./', index_name)
    return index


def get_candidate_documents_by_all_tf_scores(query, words_list, all_pls):
    """
    Calculate term frequency for each unique term in query given, 
    then sort documents by term frequencies.
    
    Args:
        query: string: query to get terms from.
        words_list: list of words in corpus.
        all_pls: list of posting lists of corpus.

    Returns:
        list of (doc_id, tf) pairs sorted by accumulated tf for doc_id. 
    """
    tf_for_doc_dict = {}
    unique_query_tokens = np.unique(query)
    set_words = set(words_list)

    for token in unique_query_tokens:  # loop all the tokens in query
        if token in set_words:  # if token in our vocabulary then lets get its pls
            token_index = words_list.index(token)  # find index of curr token to get its pls
            pls_for_curr_token = all_pls[token_index]  # gets curr token pls
            for doc_id, token_tf in pls_for_curr_token:  # loop every tuple in curr pls
                tf_for_doc_dict[doc_id] = tf_for_doc_dict.get(doc_id, 0) + token_tf  # accumulate score

    list_tf_for_doc = [(id, tf) for id, tf in tf_for_doc_dict.items()]  # split dict to tuple list
    get_tf = lambda tup: tup[1]  # sort by term freq
    sorted_docs_by_tf = sorted(list_tf_for_doc, key=get_tf, reverse=True)[:40]
    return sorted_docs_by_tf


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.
    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.
    Parameters:
    -----------
    query: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.
    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    counter = Counter(query_to_search)
    unique = np.unique(query_to_search)
    Q = np.zeros(len(unique))

    for token in unique:
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log((len(index.doclen)) / (df + epsilon), 10)  # smoothing
            try:
                # ind = unique.index(token)
                find_idx = np.where(unique == token)
                ind = find_idx[0][0]
                Q[ind] = tf * idf
            except:
                pass

    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. 
    This function will go through every token in query and fetch the corresponding information
    (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidate_docid_list.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.
    words,pls: iterator for working with posting.
    
    Returns:
    -----------
    dictionary of candidate_docid_list. In the following format: key is a pair (doc_id,term) and the value is tfidf score
    """
    candidates = {}
    N = len(index.doclen)  # for the idf calculation
    unique_terms_in_query = np.unique(query_to_search)
    for term in unique_terms_in_query:
        if term in words:
            list_of_doc = pls[words.index(term)]  # get posting list of this term
            normalized_tfidf = []
            for item in list_of_doc:  # [(id1, 5), (id2, 3) , ......]
                doc_id, freq = item[0], item[1]
                curr_document_length = index.doclen[doc_id]
                # check len: make sure we dont divide by zero
                if curr_document_length == 0:
                    zero_tf = (doc_id, 0)  # todo: check if we need to append the zero_tf ?
                    normalized_tfidf.append(zero_tf)
                else:  # calculate tf_idf, taken from lecture2 slide 27
                    df = index.df[term]  # number of documents containing this term
                    normalized_tf = freq / curr_document_length  # lecture2 slide 24
                    idf = math.log10(N / df)  # lecture2 slide 25
                    tf_idf = normalized_tf * idf  # lecture2 slide 27
                    normalized_tfidf.append((doc_id, tf_idf))

            for doc_id, tfidf in normalized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidate_docid_list for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.
    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words, pls)
    # We do not need to utilize all document. Only the documents which have terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])  # get candidate_docid_list doc id

    query_len_only_unq = len(np.unique(query_to_search))  # get len of query for D matrix cols
    unique_candidates_len = len(unique_candidates)

    D = np.zeros((unique_candidates_len, query_len_only_unq))  # get len for D matrix rows
    D = pd.DataFrame(D)  # D will be size of Number0fCandidateDocs X QueryTerms
    D.index = unique_candidates
    D.columns = np.unique(query_to_search)
    # Build the D matrix  lecture 2 slide 20
    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q, index):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.
    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows: key: document id (e.g., doc_id) value: cosine similarity score.
    """
    # Q = 1 X n matrix where 1 because one query and n is the number of tokens
    D_transpose = np.transpose(D)
    dj_q = np.dot(Q, D_transpose)
    norm_q = np.linalg.norm(Q)
    cosine_numerator = dj_q  # it's a list of values : dot product
    cosine_denominator = norm_q  # for now in the denominator there is only norm of the query
    all_cosine = cosine_numerator / cosine_denominator  # this is not the final cosine score
    cosine_dict = {}
    for idx, doc_id in enumerate(D.index):  # idx =[0,1,2,..] doc_id=[id1, id2,....]
        norm_dj = index.doc_id_2_norm.get(doc_id, 1)  # get norm from the dict
        cosine_dict[doc_id] = all_cosine[idx] * (1 / norm_dj)  # now the cosine is complete
    return cosine_dict


def get_top_n(sim_dict, N=40):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
        key: document id (e.g., doc_id)
        value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))
    N: Integer (how many documents to retrieve). By default N = 100
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    fetch_score = lambda item: item[1]
    topN = sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=fetch_score, reverse=True)[:N]
    return topN


def get_topN_score_for_query(query, index, words_list, pls, N=None):
    """
   Return the highest N documents according to the cosine similarity score.
   Generate a dictionary of cosine similarity scores.

   Parameters:
   -----------
   query: string: query to retrieve documents for.
   index: InvertedIndex: the index of a part of the corpus.
   words_list: list of words in corpus.
   pls: posting lists of corpus.
   N: Integer (how many documents to retrieve). By default, N = 100.

   Returns:
   -----------
   a ranked list of pairs (doc_id, score) in the length of N.
   """
    if N is None:
        N = 40  # 100
    query_tfidf_vector = generate_query_tfidf_vector(query, index)
    doc_term_matrix_tfidf = generate_document_tfidf_matrix(query, index, words_list, pls)
    cosine_sim = cosine_similarity(doc_term_matrix_tfidf, query_tfidf_vector, index)
    return get_top_n(cosine_sim, N)


def calc_AVGDL(index):
    """
    Helper function to calculate the average length of a document in the corpus.

    Args:
        index: InvertedIndex: the index of a part of the corpus.

    Returns: float: average document length in corpus.
    """
    return np.mean(list(index.doclen.values()))


###################### Settings ######################


BUCKET_NAME = "206201667_316399773_bucket"
PROJECT_ID = "OmriShaharIRProject"
client = storage.Client(PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# Get PageViews and PageRanks from bucket.
page_rank_file, page_views_file = "page_rank_normalized.pkl", "page_views_counter.pkl"
doc_page_rank_dict = get_content_from_storage(bucket, page_rank_file)
doc_page_view_dict = get_content_from_storage(bucket, page_views_file)

# Create Inverted Indexes from bucket.
inverted_index_text = get_index_from_bucket(bucket, 'index_body')
inverted_index_title = get_index_from_bucket(bucket, 'index_title')
inverted_index_anchor = get_index_from_bucket(bucket, 'index_anchor')

# Calculate average document length for body.
AVGDL_text = calc_AVGDL(inverted_index_text)


@app.route("/search")
def search():
    """
    Returns up to a 100 of your best search results for the query. This is
    the place to put forward your best search engine, and you are free to
    implement the retrieval whoever you'd like within the bound of the
    project requirements (efficiency, quality, etc.). That means it is up to
    you to decide on whether to use stemming, remove stopwords, use
    PageRank, query expansion, etc.

    To issue a query navigate to a URL like:
     http://YOUR_SERVER_DOMAIN/search?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
    list of up to 100 search results, ordered from best to worst where each
    element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)  # split query to tokens
    if len(query_tokens) == 0:
        return jsonify(res)

    # Get scores of Title part of corpus.
    bin_directory_title = 'postings_gcp/index_title'
    words_title, pls_title = zip(*inverted_index_title.posting_lists_iter(bin_directory_title, query_tokens))
    title_score_values = dict(get_candidate_documents_by_all_tf_scores(query_tokens, words_title, pls_title))
    # if len(query_tokens) == 1:
    #     doc_to_score = sorted([(doc_id, score) for doc_id, score in title_score_values.items()], key=lambda x: x[1], reverse=True)[:20]
    #     for doc_id, score in doc_to_score:
    #         res.append((int(doc_id), inverted_index_title.doc_id_title.get(doc_id, "")))
    #     return jsonify(res)

    # Get BM25 scores of Body (text) part of corpus.
    bin_directory_body = 'postings_gcp/index_body'
    words_text, pls_text = zip(*inverted_index_text.posting_lists_iter(bin_directory_body, query_tokens))
    bm25_body = BM25.BM_25_from_index(inverted_index_text, avg_doclen=AVGDL_text)
    bm25_scores = bm25_body.search(query_tokens, 40, words_text, pls_text)

    # Get scores of Anchor part of corpus.
    bin_directory_anchor = 'postings_gcp/index_anchor'
    words_anchor, pls_anchor = zip(*inverted_index_anchor.posting_lists_iter(bin_directory_anchor, query))
    # sorted_docs_list_anchor = get_documents_by_content(query, words_anchor, pls_anchor)
    anchor_sorted_docs_tf_list = get_candidate_documents_by_all_tf_scores(query, words_anchor, pls_anchor)
    anchor_score_values = dict(anchor_sorted_docs_tf_list[:40])
    # anchor_values = search_anchor_func(query_tokens)

    doc_to_score = defaultdict(int)
    # Use BM25 and PageViews scores for ranking.
    for doc_id, bm25score in bm25_scores:
        page_view_by_doc = doc_page_view_dict.get(doc_id, 1)
        doc_to_score[doc_id] += (3 * bm25score * page_view_by_doc) / (bm25score + page_view_by_doc)
        # page_rank_by_doc = doc_page_rank_dict.get(doc_id, 1)
        # doc_to_score[doc_id] += (3 * bm25score * page_view_by_doc) / (bm25score + page_view_by_doc)
        # doc_to_score[doc_id] += (3 * bm25score + 2.5 * page_rank_by_doc + 2 * page_view_by_doc)

    # Add title and anchor scores for ranking.
    anchor_weight = 2
    title_weight = 3
    for doc, score in title_score_values.items():
        doc_to_score[doc] *= title_weight * score
    for doc, score in anchor_score_values.items():
        doc_to_score[doc] *= anchor_weight * score

    # Sort documents by calculated score.
    doc_to_score = sorted([(doc_id, score) for doc_id, score in doc_to_score.items()], key=lambda x: x[1],
                          reverse=True)[:20]

    # Create result list of (doc_id, title) pairs.
    for doc_id, score in doc_to_score:
        res.append((int(doc_id), inverted_index_title.doc_id_title.get(doc_id, "")))

    # END SOLUTION
    return jsonify(res)


# # TODO: CHANGE NAAMA - ours is get_candidate_documents_by_all_tf_scores
# def get_documents_by_content(query_to_search, words, pls):
#     """
#     Returns ALL (not just top 100) documents that contain A QUERY WORD
#     IN THE TITLE of the article, ordered in descending order of the NUMBER OF
#     QUERY WORDS that appear in the text.
#     return : list of (doc id, number of words) , sort in descending order
#     """
#     candidates = {}
#     for term in np.unique(query_to_search):
#         if term in words:
#             list_of_doc = pls[words.index(term)]
#             for doc_item in list_of_doc:
#                 candidates[doc_item[0]] = candidates.get(doc_item[0], 0) + doc_item[1]
#     return sorted(candidates.items(), key=lambda item: item[1], reverse=True)


# # TODO: CHANGE NAAMA - put inside search func
# def search_anchor_func(query):
#     bin_directory = 'postings_gcp/index_anchor'
#     words_anchor, pls_anchor = zip(*inverted_index_anchor.posting_lists_iter(bin_directory, query))
#     sorted_docs_list = get_documents_by_content(query, words_anchor, pls_anchor)
#     return dict(sorted_docs_list[:20])


@app.route("/search_body")
def search_body():
    """
    Returns up to a 100 search results for the query using TFIDF AND COSINE
    SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
    staff-provided tokenizer from Assignment 3 (GCP part) to do the
    tokenization and remove stopwords.

    To issue a query navigate to a URL like:
     http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
    list of up to 100 search results, ordered from best to worst where each
    element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)  # split query to tokens
    bin_directory = 'postings_gcp/index_body'
    words_text, pls_text = zip(*inverted_index_text.posting_lists_iter(bin_directory, query_tokens))
    # print(f"search_body - if this is printed then zip worked")
    # ['w1','w2'], [pls of w1, pls of w2]
    top_n_docs_for_query = get_topN_score_for_query(query_tokens, inverted_index_text, words_text, pls_text)
    for doc_id, score in top_n_docs_for_query:
        res.append((int(doc_id), inverted_index_text.doc_id_title[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """
    Returns ALL (not just top 100) search results that contain A QUERY WORD
    IN THE TITLE of articles, ordered in descending order of the NUMBER OF
    DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
    USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
    tokenization and remove stopwords. For example, a document
    with a title that matches two distinct query words will be ranked before a
    document with a title that matches only one distinct query word,
    regardless of the number of times the term appeared in the title (or
    query).

    Test this by navigating to the a URL like:
    http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
    list of ALL (not just top 100) search results, ordered from best to
    worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)  # split query to tokens
    bin_directory = 'postings_gcp/index_title'
    words_title, pls_title = zip(*inverted_index_title.posting_lists_iter(bin_directory, query_tokens))
    # ['title1','title2'], [pls of title1, pls of title2]
    candidate_docs_sorted_by_tf = get_candidate_documents_by_all_tf_scores(query_tokens, words_title, pls_title)
    for id_tf in candidate_docs_sorted_by_tf:
        curr_doc_id, token_tf = int(id_tf[0]), id_tf[1]
        curr_title = inverted_index_title.doc_id_title[curr_doc_id]
        res.append((curr_doc_id, curr_title))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """
    Returns ALL (not just top 100) search results that contain A QUERY WORD
    IN THE ANCHOR TEXT of articles, ordered in descending order of the
    NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
    DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
    3 (GCP part) to do the tokenization and remove stopwords. For example,
    a document with a anchor text that matches two distinct query words will
    be ranked before a document with anchor text that matches only one
    distinct query word, regardless of the number of times the term appeared
    in the anchor text (or query).

    Test this by navigating to the a URL like:
    http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    bin_directory = 'postings_gcp/index_anchor'
    words_anchor, pls_anchor = zip(*inverted_index_anchor.posting_lists_iter(bin_directory, query_tokens))
    candidate_docs_sorted_by_tf = get_candidate_documents_by_all_tf_scores(query_tokens, words_anchor, pls_anchor)
    for id_tf in candidate_docs_sorted_by_tf:
        curr_doc_id, token_tf = int(id_tf[0]), id_tf[1]
        curr_title = inverted_index_anchor.doc_id_title.get(curr_doc_id, "")
        res.append((curr_doc_id, curr_title))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """
    Returns PageRank values for a list of provided wiki article IDs.

    Test this by issuing a POST request to a URL like:
      http://YOUR_SERVER_DOMAIN/get_pagerank
    with a json payload of the list of article ids. In python do:
      import requests
      requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
    As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
    list of floats:
      list of PageRank scores that correrspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        rank = float(doc_page_rank_dict.get(wiki_id, 0))
        res.append(rank)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """
    Returns the number of page views that each of the provided wiki articles
    had in August 2021.

    Test this by issuing a POST request to a URL like:
      http://YOUR_SERVER_DOMAIN/get_pageview
    with a json payload of the list of article ids. In python do:
      import requests
      requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
    As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
    list of ints:
      list of page view numbers from August ;21 that correrspond to the
      provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        view_score = doc_page_view_dict.get(wiki_id, 0)
        res.append(view_score)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
