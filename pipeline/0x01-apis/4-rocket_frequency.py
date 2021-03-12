#!/usr/bin/env python3
""" script that displays the number of launches per rocket """
import requests

if __name__ == '__main__':
    rockets = {}
    r = requests.get('https://api.spacexdata.com/v4/launches')
    if r.status_code == 200:
        launches = r.json()
        for launch in launches:
            rocket_id = launch['rocket']
            r = requests.get('https://api.spacexdata.com/v4/rockets/{}'.
                             format(rocket_id))
            if r.status_code == 200:
                rocket_name = r.json()['name']
                if rocket_name in rockets.keys():
                    rockets[rocket_name] += 1
                else:
                    rockets[rocket_name] = 1
        rockets = sorted(rockets.items(), key=lambda kv: kv[0])
        rockets = sorted(rockets, key=lambda kv: kv[1], reverse=True)
        for rocket in rockets:
            print(*rocket, sep=": ")
