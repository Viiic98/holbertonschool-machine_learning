#!/usr/bin/env python3
""" script that displays the upcoming
    launch with these information """
import requests
import time

if __name__ == '__main__':
    now = time.time()
    r = requests.get("https://api.spacexdata.com/v4/launches/upcoming")
    if r.status_code == 200:
        launches = r.json()
        date = 0
        for i, launch in enumerate(launches):
            if launch['date_unix'] > date:
                if launch['date_unix'] > now and date != 0:
                    break
                date = launch['date_unix']
                idx = i
        launch_name = launches[idx]['name']
        date = launches[idx]['date_local']
        rocket_id = launches[idx]['rocket']
        r = requests.get('https://api.spacexdata.com/v4/rockets/{}'.
                         format(rocket_id))
        if r.status_code == 200:
            rocket_name = r.json()['name']
            pad_id = launches[idx]['launchpad']
            r = requests.get('https://api.spacexdata.com/v4/launchpads/{}'.
                             format(pad_id))
            if r.status_code == 200:
                pad_data = r.json()
                pad_name = pad_data['name']
                pad_locality = pad_data['locality']
    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name,
                                        pad_name, pad_locality))
