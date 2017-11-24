#-*- coding: utf-8 -*-

from flask import Flask
import pymysql
import json

app = Flask(__name__)

conn = pymysql.connect(host = 'localhost', user = 'root', password = 'digh3484', db = 'handy_tutor', charset = 'utf8')


@app.route('/video_list')
def getVideoList():
	cursor = conn.cursor(pymysql.cursors.DictCursor)
	cursor.execute('SELECT * FROM VIDEO');
	rows = cursor.fetchall()
	return json.dumps(rows)


if __name__ == '__main__':
	app.run(host = '0.0.0.0', port=5000, debug=True)
