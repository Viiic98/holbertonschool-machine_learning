#!/usr/bin/env python3
""" script that displays the upcoming
    launch with these information """
import requests
import time

if __name__ == '__main__':
    now = time.time()
    r = requests.get("https://api.spacexdata.com/v4/launches/upcoming")
    if r.status_code == 200:
        launches = sorted(r.json(), key=lambda i: i['date_unix'])
        launch_name = launches[0]['name']
        date = launches[0]['date_local']
        rocket_id = launches[0]['rocket']
        r = requests.get('https://api.spacexdata.com/v4/rockets/{}'.
                         format(rocket_id))
        if r.status_code == 200:
            rocket_name = r.json()['name']
            pad_id = launches[0]['launchpad']
            r = requests.get('https://api.spacexdata.com/v4/launchpads/{}'.
                             format(pad_id))
            if r.status_code == 200:
                pad_data = r.json()
                pad_name = pad_data['name']
                pad_locality = pad_data['locality']
    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name,
                                        pad_name, pad_locality))
