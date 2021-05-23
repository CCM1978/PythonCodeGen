'''import json
import requests
import urllib3

def test():
    url = "https://mapps.cricbuzz.com/cbzios/match/livematches"
    http = urllib3.PoolManager()
    #url = 'http://webcode.me'
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = { 'User-Agent' : user_agent }
    resp = http.request('GET', url,headers)
    return resp.data'''
#id=matches[0]['id']
###print(liveMatches())
#lscore = c.commentary('29548')
#print(json.dumps(lscore, indent=4, sort_keys=True))
#import gunicorn
