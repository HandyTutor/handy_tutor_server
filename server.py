#-- coding: utf-8 --

from flask import Flask
from flask import request
import pymysql
import json
from PhraseSimilarity import SimilarityScore

app = Flask(__name__)

conn = pymysql.connect(host = 'localhost', user = 'root', password = 'digh3484', db = 'handy_tutor', charset = 'utf8')


@app.route('/video_list')
def getVideoList():
	cursor = conn.cursor(pymysql.cursors.DictCursor)
	cursor.execute('SELECT * FROM VIDEO');
	rows = cursor.fetchall()

	return json.dumps(rows)

@app.route('/video', methods = ['POST'])
def getVideo():

	cursor = conn.cursor(pymysql.cursors.DictCursor)
	cursor.execute('SELECT KOREAN, ENGLISH, ROLE, TIME FROM SUBTITLE WHERE VIDEO_KEY = "' + request.form['video_key'] + '" ORDER BY TIME ASC');
	rows = cursor.fetchall()
	return json.dumps(rows)

@app.route('/phrase_similarity', methods = ['POST'])
def getSimilarity():
	return str(SimilarityScore(request.form['input1'], request.form['input2']))
	

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port=5002, debug=True)
