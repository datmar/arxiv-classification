from config import *

import sys
import urllib.request
import urllib.parse
import json
import pprint
import pandas as pd
import pickle
from pathlib import Path


"""
This api requestor was altered from
https://github.com/ronentk/sci-paper-miner/blob/master/core_requestor.py,
which was in turn adapted from
https://github.com/oacore/or2016-api-demo/blob/master/OR2016%20API%20demo.ipynb
"""


class CoreApiRequestor:

    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key
        #defaults
        self.pagesize = 100
        self.page = 1

    def parse_response(self, decoded):
        res = []
        for item in decoded['data']:
            doi = None
            if 'identifiers' in item:
                for identifier in item['identifiers']:
                    if identifier and identifier.startswith('doi:'):
                        doi = identifier
                        break
            res.append([item['title'], doi])
        return res

    def request_url(self, url):
        with urllib.request.urlopen(url) as response:
            html = response.read()
        return html

    def get_method_query_request_url(self,method,query,fullText,page):
        if (fullText):
            fullText = 'true'
        else:
            fullText = 'false'
        params = {
            'apiKey':self.api_key,
            'page':page,
            'pageSize':self.pagesize,
            'fulltext':fullText
        }
        return self.endpoint + method + '/' + urllib.parse.quote(query) + '?' + urllib.parse.urlencode(params)

endpoint = 'https://core.ac.uk/api-v2'

# Request an api key at https://core.ac.uk/api-keys/register/
# Create a config.py file in root directory and assign API_KEY = "your_api_key"

api_key = API_KEY

method = '/articles/search'
api = CoreApiRequestor(endpoint,api_key)

my_file = Path("./data/two-topic-fulltext.pkl")

if my_file.is_file():
    pass
else:
    topics = ['deep AND learning', 'computer AND vision']

    topic_results = [[], []]
    for i, topic in enumerate(topics):
        for page in range(1, 101):
            '''
            Get url
            '''
            url = api.get_method_query_request_url(method,topic,True,page)
            url

            '''
            Get results
            '''
            response = api.request_url(url)
            result = json.loads(response.decode('utf-8'))
            topic_results[i].append(result["data"])

    d = []
    for i, topic in enumerate(topics):
        for page in topic_results[i]:
            for article in page:
                d.append({'topic': i, 'full_text': article['fullText']})
    df = pd.DataFrame(d)

    # pickle dataframe for future use
    df.to_pickle('./data/two-topic-fulltext.pkl')


