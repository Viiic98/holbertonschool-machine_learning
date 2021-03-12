#!/usr/bin/env python3
""" Passengers """
import requests


def availableShips(passengerCount):
    """ returns the list of ships that can hold a
        given number of passengers """
    total_ships = []
    next = "https://swapi-api.hbtn.io/api/starships/"
    while next:
        r = requests.get(next)
        if r.status_code != 200:
            break
        data = r.json()
        ships = data['results']
        for ship in ships:
            passengers = ship['passengers'].replace(',', '')
            try:
                passengers = int(passengers)
            except Exception as e:
                passengers = 0
            if passengers >= passengerCount:
                total_ships.append(ship['name'])
        next = data['next']
    return total_ships
