# coding: UTF-8
from requests_oauthlib import OAuth1Session
import settings #api接続情報
import config #db接続情報

# テーブル作成
# apiからデータ取得
def get_data():
	# apiに接続するためのデータを格納
	consumer_key = settings.consumer_key
	consumer_secret = settings.consumer_secret
	access_token = settings.access_token
	access_token_secret = settings.access_token_secret

	# oauthを用いて接続
	angel = OAuth1Session(
		consumer_key = consumer_key, 
		consumer_secret = consumer_secret, 
		access_token = access_token, 
		access_token_secret = access_token_secret)

	request = angel.get("https://angel.co/api/startups/:id")

	# angel_listの場合はjsonで返ってくるから以下のコードは不要?
	# data = request.json()

	# api接続を閉じる?<なぜ必要かわからず>
	session.close()

# dbにinsert
def dbinsert():
	# db接続
	dbcon = mysql.connector.connect(database=config.db, user=config.user, password=config.passwd, host=config.host)
	dbcur = dbcon.cursor()

	insert = "insert into "
	dbcur.execute(insert)



if __name__ == '__main__':
	main()