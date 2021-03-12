#!/usr/bin/env python3
""" script that prints the location of a specific user """
import requests
import time


if __name__ == '__main__':
    user = 'https://api.github.com/users/holbertonschool'
    r = requests.get(user)
    if r.status_code == 200:
        print(r.json()['location'])
    elif r.status_code == 403:
        limit = r.headers['X-Ratelimit-Reset']
        limit = int((int(limit) - int(time.time())) / 60)
        print('Reset in {} min'.format(limit))
    else:
        print("Not found")
