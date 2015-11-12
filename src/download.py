# -*- coding=utf-8 -*-

from requests import Request, Session
import os
import json
import shutil
import logging
import time
import random
from termcolor import colored

if 'data' not in os.listdir('.'):
	os.mkdir('data')
os.chdir('data')

s = Session()
HEADERS = {
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
	'Accept-Encoding': 'gzip, deflate, sdch',
	'Accept-Language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4',
	'Cache-Control': 'max-age=0',
	'Connection': 'keep-alive',
	'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36'
}
sleep_time = 2

def _get(url):
        while True:
		try:
			ret = s.get(url, headers=HEADERS, stream=True)
			if '"code": 404' in ret.content:
				continue
			return ret
		except Exception, e:
			logger.error(e)
			time.sleep(0.5 + random.random())

def get_captcha_info():
	url = 'http://www.douban.com/j/misc/captcha'
	r = s.get(url, headers=HEADERS, timeout=5)
	logger.debug(r.content)
	info = json.loads(r.content)
	info['url'] = 'http:' + info['url']
	time.sleep(0.1)
	return info

def download_captcha_image():
	info = get_captcha_info()
	r = s.get(info['url'], headers=HEADERS, stream=True, timeout=5)
	img_raw = r.raw
	file_name = info['token'][:-3] + '.jpg'
	if file_name not in os.listdir('.'):
		with open(file_name, 'wb') as fImage:
			shutil.copyfileobj(img_raw, fImage)
		logger.info('Image %s - %.2f KB' % (file_name, float(r.headers['Content-Length'])/1024 ))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(filename)s: %(lineno)s [%(message)s]')
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('downloader')
while True:
	try:
		download_captcha_image()
		sleep_time = 2
	except Exception, e:
		logger.exception('Wow... sleep for %d s' % sleep_time)
		s.cookies.clear()
		time.sleep(sleep_time)
		sleep_time *= 2
	time.sleep(1 + 1*random.random())
