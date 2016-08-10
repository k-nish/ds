#encoding: UTF-8
from requests_oauthlib import OAuth1Session
import json
import settings
import urllib2, sys
import config

twitter = OAuth1Session(settings.CONSUMER_KEY, settings.CONSUMER_SECRET, settings.ACCESS_TOKEN, settings.ACCESS_TOKEN_SECRET)

try: citycode = sys.argv[1]
except: citycode = '130010' #デフォルト地域を東京にする
resp = urllib2.urlopen('http://weather.livedoor.com/forecast/webservice/json/v1?city=%s'%citycode).read()

resp = json.loads(resp)

result = u"東京の天気"
for forecast in resp['forecasts']:
	result = result + "\n" + forecast['dateLabel']+'('+forecast['date']+')'+forecast['telop']

params = {"status": result}
req = twitter.post("https://api.twitter.com/1.1/statuses/update.json",params = params)

# api接続
# データ取得
# dbにテーブル作成
# dbにinsert


# レスポンスを確認
if req.status_code == 200:
    print ("OK")
else:
    print ("Error: %d" % req.status_code)
