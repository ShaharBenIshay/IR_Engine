import math
import numpy as np
import pandas as pd


def get_top_n(sim_dict, N=40):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
        key: document id (e.g., doc_id)
        value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))
    N: Integer (how many documents to retrieve). By default N = 3
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    fetch_score = lambda item: item[1]
    topN = sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=fetch_score, reverse=True)[:N]
    return topN


class BM_25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.2

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, avg_doclen, k1=1.2, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.doclen)
        self.idf = None
        self.AVGDL = avg_doclen

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: BM25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                # Notice: usage of log2 instead of log
                idf[term] = math.log2(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_documents_list(self, query, words, pls):
        """
        Generate a list representing a pool of candidate documents for a given query.
        This function will go through every token in query and populate the list 'candidate_docid_list'.

        Parameters:
        -----------
        query: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
        index: inverted index loaded from the corresponding files.
        words,pls: generator for working with posting.

        Returns:
        -----------
        list of candidate_docid_list doc_ids.
        """
        candidates_list = []
        words_unq = set(words)
        for term in np.unique(query):
            if term in words_unq:
                term_list_of_doc = (pls[words.index(term)])
                for docid_tf in term_list_of_doc:
                    cand_doc_id = docid_tf[0]
                    candidates_list.append(cand_doc_id)
        return np.unique(candidates_list)

    def search(self, query, N=40, query_words=None, query_pls=None):
        """
        This function calculate the BM25 score for given query and document.
        We need to check only documents which are 'candidate_docid_list' for a given query.
        This function return a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        N: integer: number of relevant documents to return.

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """
        candidate_docs_list = self.get_candidate_documents_list(query, query_words, query_pls)
        self.idf = self.calc_idf(query)  # need to create idf attr for this class
        scores_dict = self._score(query, candidate_docs_list, query_words, query_pls)
        return get_top_n(scores_dict, N)

    def calc_bm25_formula(self, doc_id, term_frequencies, idf):
        """
        This function calculate the BM25 score by its formula.

        Parameters:
        -----------
        doc_id: current document id.
        term_frequencies: dictionary of term frequencies by document ids.
        idf: idf value of current document id.

        Returns:
        -----------
        float: BM25 score.
        """
        doc_len = self.index.doclen.get(doc_id, 0)
        doc_freq = term_frequencies.get(doc_id, 0)
        bm25_numerator = idf * doc_freq * (self.k1 + 1)
        bm25_denominator = doc_freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
        return bm25_numerator / bm25_denominator

    def _score(self, query, candidate_docid_list, query_words=None, query_pls=None):
        """
        This function calculate the BM25 score for given query and candidate documents.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        candidate_docid_list: list of doc_ids.

        Returns:
        -----------
        dictionary of scores: float, BM25 score by doc_ids.
        """
        df_docid_scores = pd.DataFrame({'doc_id': candidate_docid_list})
        df_docid_scores["score"] = 0
        # TODO: term_total
        # print(f"term_total len: {len(self.index.term_total)}")
        for term in query:
            try:
                # if term in self.index.term_total.keys():
                query_term_freq = dict(query_pls[query_words.index(term)])
                idf = self.idf.get(term, 0)
                df_docid_scores["score"] += df_docid_scores["doc_id"].apply(lambda doc_id: self.calc_bm25_formula(
                                                                            doc_id, query_term_freq, idf))
            except:
                # print("DIDN'T FIND TERM IN QUERY / TERM TOTAL")
                pass

        scores_values = df_docid_scores["score"].values
        df_dict_index_docid = df_docid_scores["doc_id"]
        doc_scores_dict = pd.Series(scores_values, index=df_dict_index_docid).to_dict()
        return doc_scores_dict
