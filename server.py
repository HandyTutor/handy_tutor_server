#-- coding: utf-8 --

from flask import Flask
from flask import request
import pymysql
import json

app = Flask(__name__)

conn = pymysql.connect(host = 'localhost', user = 'root', password = 'digh3484', db = 'handy_tutor', charset = 'utf8')


@app.route('/video_list')
def getVideoList():
	cursor = conn.cursor(pymysql.cursors.DictCursor)
	cursor.execute('SELECT * FROM VIDEO');
	rows = cursor.fetchall()

	return json.dumps(rows, encoding = 'utf-8')

@app.route('/video', methods = ['POST'])
def getVideo():

	cursor = conn.cursor(pymysql.cursors.DictCursor)
	cursor.execute('SELECT KOREAN, ENGLISH, ROLE, TIME FROM SUBTITLE WHERE VIDEO_KEY = "' + request.form['video_key'] + '"');
	rows = cursor.fetchall()
	return json.dumps(rows, encoding = 'utf-8')

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port=5000, debug=True)
