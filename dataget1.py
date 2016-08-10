#encoding: UTF-8
from requests_oauthlib import OAuth1Session
import json
import urllib2, sys
import settings
import config

# 以下twitter用のコード
# twitter = OAuth1Session(settings.CONSUMER_KEY, settings.CONSUMER_SECRET, settings.ACCESS_TOKEN, settings.ACCESS_TOKEN_SECRET)
# try: citycode = sys.argv[1]
# except: citycode = '130010' #デフォルト地域を東京にする
# resp = urllib2.urlopen('http://weather.livedoor.com/forecast/webservice/json/v1?city=%s'%citycode).read()
# resp = json.loads(resp)
# result = u"東京の天気"
# for forecast in resp['forecasts']:
# 	result = result + "\n" + forecast['dateLabel']+'('+forecast['date']+')'+forecast['telop']
# params = {"status": result}
# req = twitter.post("https://api.twitter.com/1.1/statuses/update.json",params = params)


angel = OAuth1Session()
# api接続
response = angel.get('情報を取得するapiのurl')

# db接続
dbcon = mysql.connector.connect(database=config.db, user=config.user, password=config.passwd, host=config.host)
dbcur = dbcon.cursor()

# dbにテーブル作成
# まず作りたいテーブルの名前が使われていたらそのテーブルを消去
drop = "drop table if exists <テーブル名>;"
dbcur.execute(drop)
# テーブルを作成する
create = "create table <テーブル名>~"
dbcur.execute(create)
# dbにinsert
insert = "insert into "
dbcur.execute(insert)

dbcon.commit()
# カーソルを閉じる
dbcur.close()
# dbを閉じる
dbcon.close()

# レスポンスを確認
if req.status_code == 200:
    print ("OK")
else:
    print ("Error: %d" % req.status_code)
